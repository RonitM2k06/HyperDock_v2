from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Date, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Optional
from datetime import date
import csv
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

DATABASE_URL = "sqlite:///./test.db"  # Consider environment variables for production

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    itemId = Column(String, index=True, unique=True)
    name = Column(String, index=True)
    width = Column(Integer, nullable=False)
    depth = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    mass = Column(Float, nullable=False)  # Use Float for mass
    priority = Column(Integer, index=True, nullable=False)
    expiryDate = Column(Date)
    usageLimit = Column(Integer, nullable=False, default=0)  # Default usage
    preferredZone = Column(String, nullable=False)


class Container(Base):
    __tablename__ = "containers"
    id = Column(Integer, primary_key=True, index=True)
    containerId = Column(String, index=True, unique=True)
    zone = Column(String, index=True, nullable=False)
    width = Column(Integer, nullable=False)
    depth = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict origins in production!
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()  # Rollback on any exception
        raise e
    finally:
        db.close()


@app.get("/")
def home():
    return {"message": "Space Cargo API is running!", "frontend": "/static/index.html"}


# API Models
class ItemSchema(BaseModel):
    itemId: str = Field(..., example="item001")
    name: str = Field(..., example="Water Bottle")
    width: int = Field(..., example=10)
    depth: int = Field(..., example=10)
    height: int = Field(..., example=20)
    mass: float = Field(..., example=0.5)
    priority: int = Field(..., example=1)
    expiryDate: Optional[date] = Field(None, example="2025-12-25")
    usageLimit: int = Field(..., example=50)
    preferredZone: str = Field(..., example="ZoneA")


class ContainerSchema(BaseModel):
    containerId: str = Field(..., example="container001")
    zone: str = Field(..., example="ZoneA")
    width: int = Field(..., example=100)
    depth: int = Field(..., example=100)
    height: int = Field(..., example=100)


class PlacementRequest(BaseModel):
    items: List[ItemSchema]
    containers: List[ContainerSchema]


class PlacementResponse(BaseModel):
    success: bool
    placements: List[dict]
    rearrangements: List[dict]


class SearchResponse(BaseModel):
    success: bool
    found: bool
    item: Optional[dict]
    retrievalSteps: Optional[List[dict]]


class RetrieveRequest(BaseModel):
    itemId: str = Field(..., example="item001")
    userId: str = Field(..., example="astronaut1")
    timestamp: str = Field(..., example="2025-03-15T10:00:00")


class PlaceRequest(BaseModel):
    itemId: str = Field(..., example="item001")
    userId: str = Field(..., example="astronaut1")
    timestamp: str = Field(..., example="2025-03-15T10:00:00")
    containerId: str = Field(..., example="container001")
    position: dict = Field(..., example={"startCoordinates": {"width": 0, "depth": 0, "height": 0}, "endCoordinates": {"width": 10, "depth": 10, "height": 20}})


class WasteIdentifyResponse(BaseModel):
    success: bool
    wasteItems: List[dict]


class WasteReturnPlanRequest(BaseModel):
    undockingContainerId: str = Field(..., example="container001")
    undockingDate: str = Field(..., example="2025-04-01")
    maxWeight: float = Field(..., example=100.0)


class WasteReturnPlanResponse(BaseModel):
    success: bool
    returnPlan: List[dict]
    retrievalSteps: List[dict]
    returnManifest: dict


class CompleteUndockingRequest(BaseModel):
    undockingContainerId: str = Field(..., example="container001")
    timestamp: str = Field(..., example="2025-04-02T12:00:00")


class ApiResponse(BaseModel):
    success: bool


class TimeSimulationRequest(BaseModel):
    numOfDays: Optional[int] = Field(None, example=1)
    toTimestamp: Optional[str] = Field(None, example="2025-04-03T00:00:00")
    itemsToBeUsedPerDay: List[dict] = Field(..., example=[{"itemId": "item002"}])


class TimeSimulationResponse(BaseModel):
    success: bool
    newDate: str = Field(..., example="2025-04-02")
    changes: dict = Field(..., example={"itemsUsed": [{"itemId": "item002", "name": "Test Item", "remainingUses": 49}], "itemsExpired": [], "itemsDepletedToday": []})


class ImportResponse(BaseModel):
    success: bool
    itemsImported: int = Field(..., example=10)
    errors: List[dict] = Field(..., example=[{"row": {"itemId": "bad_id"}, "message": "Invalid data"}])


class LogEntry(BaseModel):
    timestamp: str = Field(..., example="2025-03-13T10:00:00")
    userId: str = Field(..., example="astronaut1")
    actionType: str = Field(..., example="placement")
    itemId: str = Field(..., example="001")
    details: dict = Field(..., example={"fromContainer": "contA", "toContainer": "contB", "reason": "space optimization"})


from fastapi import HTTPException


@app.post("/api/containers")
def add_containers(containers: List[ContainerSchema], db: Session = Depends(get_db)):
    try:
        for container in containers:
            existing = db.query(Container).filter(Container.containerId == container.containerId).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Container {container.containerId} already exists.")

            db_container = Container(**container.dict())
            db.add(db_container)
        db.commit()
        return {"message": "Containers added successfully", "total": len(containers)}
    except Exception as e:
        logging.error(f"Error adding containers: {e}")
        raise  # Re-raise the exception to be handled by FastAPI's default error handler


# API: Add New Cargo Items
@app.post("/api/items")
def add_items(items: List[ItemSchema], db: Session = Depends(get_db)):
    try:
        for item in items:
            db_item = Item(**item.dict())
            db.add(db_item)
        db.commit()
        return {"message": "Items added successfully", "total": len(items)}
    except Exception as e:
        logging.error(f"Error adding items: {e}")
        raise


# API: Get All Containers
@app.get("/api/containers")
def get_containers(db: Session = Depends(get_db)):
    try:
        containers = db.query(Container).all()
        return {"containers": containers}
    except Exception as e:
        logging.error(f"Error getting containers: {e}")
        raise


@app.get("/api/items")
def get_items(db: Session = Depends(get_db)):
    try:
        items = db.query(Item.itemId, Item.name, Item.priority).all()  # ✅ Fetch only required columns
        return {"items": [{"itemId": i[0], "name": i[1], "priority": i[2]} for i in items]}
    except Exception as e:
        logging.error(f"Error getting items: {e}")
        raise


# API: Get a Specific Item by ID
@app.get("/api/items/{item_id}")
def get_item(item_id: str, db: Session = Depends(get_db)):
    try:
        item = db.query(Item).filter(Item.itemId == item_id).first()
        if item:
            return item
        raise HTTPException(status_code=404, detail="Item not found")  # Use HTTPException
    except Exception as e:
        logging.error(f"Error getting item: {e}")
        raise


# API: Placement Recommendations
@app.post("/api/placement", response_model=PlacementResponse)
def get_placement(req: PlacementRequest, db: Session = Depends(get_db)):
    try:
        placements = []
        rearrangements = []

        for item in req.items:
            best_container = None
            best_position = None
            min_wasted_space = float('inf')

            for container in req.containers:
                available_space = container.width * container.depth * container.height
                item_volume = item.width * item.depth * item.height

                if item_volume <= available_space and item.preferredZone == container.zone:
                    wasted_space = available_space - item_volume
                    if wasted_space < min_wasted_space:
                        min_wasted_space = wasted_space
                        best_container = container
                        best_position = {
                            "startCoordinates": {"width": 0, "depth": 0, "height": 0},
                            "endCoordinates": {"width": item.width, "depth": item.depth, "height": item.height}
                        }

            if best_container:
                placements.append({
                    "itemId": item.itemId,
                    "containerId": best_container.containerId,
                    "position": best_position
                })

        return {
            "success": True,
            "placements": placements,
            "rearrangements": rearrangements
        }
    except Exception as e:
        logging.error(f"Error getting placement: {e}")
        raise


# API: Item Search and Retrieval
@app.get("/api/search", response_model=SearchResponse)
def search_item(itemId: Optional[str] = Query(None), itemName: Optional[str] = Query(None), db: Session = Depends(get_db)):
    try:
        found = False
        item_info = {}
        retrieval_steps = []

        if itemId:
            item = db.query(Item).filter(Item.itemId == itemId).first()
        elif itemName:
            item = db.query(Item).filter(Item.name == itemName).first()
        else:
            raise HTTPException(status_code=400, detail="Either itemId or itemName must be provided")

        if item:
            found = True
            item_info = {
                "itemId": item.itemId,
                "name": item.name,
                "containerId": "contA",  #  Placeholder - You'll need real container logic
                "zone": item.preferredZone,
                "position": {  # Placeholder - You'll need real position logic
                    "startCoordinates": {"width": 0, "depth": 0, "height": 0},
                    "endCoordinates": {"width": item.width, "depth": item.depth, "height": item.height}
                }
            }

            retrieval_steps = [
                {"step": 1, "action": "retrieve", "itemId": item.itemId, "itemName": item.name}
            ]

        return {
            "success": True,
            "found": found,
            "item": item_info,
            "retrievalSteps": retrieval_steps
        }
    except Exception as e:
        logging.error(f"Error searching item: {e}")
        raise


# API: Retrieve Item
@app.post("/api/retrieve", response_model=ApiResponse)
def retrieve_item(req: RetrieveRequest, db: Session = Depends(get_db)):
    try:
        #  Implement your retrieval logic here (e.g., update database, create logs)
        #  For now, it returns success
        return {"success": True}
    except Exception as e:
        logging.error(f"Error retrieving item: {e}")
        raise


# API: Place Item
@app.post("/api/place", response_model=ApiResponse)
def place_item(req: PlaceRequest, db: Session = Depends(get_db)):
    try:
        #  Implement your placement logic here (e.g., update database, create logs)
        #  For now, it returns success
        return {"success": True}
    except Exception as e:
        logging.error(f"Error placing item: {e}")
        raise


# API: Identify Waste Items
@app.get("/api/waste/identify", response_model=WasteIdentifyResponse)
def identify_waste_items(db: Session = Depends(get_db)):
    try:
        today = datetime.now().date()
        waste_items = []
        items = db.query(Item).all()
        for item in items:
            reason = None
            if item.expiryDate and item.expiryDate < today:
                reason = "Expired"
            elif item.usageLimit == 0:
                reason = "Out of Uses"

            if reason:
                waste_items.append({
                    "itemId": item.itemId,
                    "name": item.name,
                    "reason": reason,
                    "containerId": "contA",  # Placeholder
                    "position": {  # Placeholder
                        "startCoordinates": {"width": 0, "depth": 0, "height": 0},
                        "endCoordinates": {"width": item.width, "depth": item.depth, "height": item.height}
                    }
                })

        return {
            "success": True,
            "wasteItems": waste_items
        }
    except Exception as e:
        logging.error(f"Error identifying waste: {e}")
        raise


# API: Generate Waste Return Plan
@app.post("/api/waste/return-plan", response_model=WasteReturnPlanResponse)
def generate_waste_return_plan(req: WasteReturnPlanRequest, db: Session = Depends(get_db)):
    try:
        return_plan = []
        retrieval_steps = []
        return_manifest = {
            "undockingContainerId": req.undockingContainerId,
            "undockingDate": req.undockingDate,
            "returnItems": [],
            "totalVolume": 0,
            "totalWeight": 0
        }

        # Convert undockingDate string to a datetime.date object
        undocking_date = datetime.strptime(req.undockingDate, "%Y-%m-%d").date()

        # Fetch only expired items directly from the database
        expired_items = db.query(Item).filter(Item.expiryDate != None, Item.expiryDate < undocking_date).all()

        for item in expired_items:
            return_manifest["returnItems"].append({
                "itemId": item.itemId,
                "name": item.name,
                "expiryDate": item.expiryDate.strftime("%Y-%m-%d"),
                "containerId": req.undockingContainerId  # Placeholder
            })
            return_manifest["totalVolume"] += item.width * item.depth * item.height
            return_manifest["totalWeight"] += item.mass

        return {
            "success": True,
            "returnPlan": return_plan,
            "retrievalSteps": retrieval_steps,
            "returnManifest": return_manifest
        }
    except Exception as e:
        logging.error(f"Error generating return plan: {e}")
        raise


# API: Complete Waste Undocking
@app.post("/api/waste/complete-undocking", response_model=ApiResponse)
def complete_undocking(req: CompleteUndockingRequest, db: Session = Depends(get_db)):
    try:
        #  Implement undocking completion logic here (e.g., update database, create logs)
        #  For now, it returns a placeholder value
        return {"success": True, "itemsRemoved": 5}  # Placeholder
    except Exception as e:
        logging.error(f"Error completing undocking: {e}")
        raise


# API: Simulate Time
@app.post("/api/simulate/day", response_model=TimeSimulationResponse)
def simulate_time(req: TimeSimulationRequest, db: Session = Depends(get_db)):
    try:
        today = datetime.now().date()
        new_date = None
        if req.numOfDays:
            new_date = today + timedelta(days=req.numOfDays)
        elif req.toTimestamp:
            new_date = datetime.strptime(req.toTimestamp, "%Y-%m-%dT%H:%M:%S").date()
        else:
            raise HTTPException(status_code=400, detail="Either numOfDays or toTimestamp must be provided")

        items_used = []
        items_expired = []
        items_depleted = []

        items = db.query(Item).all()
        for item in items:
            if item.expiryDate and item.expiryDate < new_date:
                items_expired.append({"itemId": item.itemId, "name": item.name})
            if item.usageLimit == 0:
                items_depleted.append({"itemId": item.itemId, "name": item.name})

        #  Simulate item usage (Placeholder - Requires actual logic)
        #  This is just a basic example:
        for used_item in req.itemsToBeUsedPerDay:
            item_from_db = db.query(Item).filter(Item.itemId == used_item["itemId"]).first()
            if item_from_db:
                item_from_db.usageLimit = max(0, item_from_db.usageLimit - 1)  #  Decrement usage
                db.commit()
                items_used.append({"itemId": item_from_db.itemId, "name": item_from_db.name, "remainingUses": item_from_db.usageLimit})

        return {
            "success": True,
            "newDate": new_date.strftime("%Y-%m-%d"),
            "changes": {
                "itemsUsed": items_used,
                "itemsExpired": items_expired,
                "itemsDepletedToday": items_depleted
            }
        }
    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        logging.error(f"Error simulating time: {e}")
        raise


# API: Import Items from CSV
@app.post("/api/import/items", response_model=ImportResponse)
def import_items(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = file.file.read().decode('utf-8')
        reader = csv.DictReader(contents.splitlines())
        items_imported = 0
        errors = []

        for row in reader:
            try:
                #  Validate data types from CSV
                item_data = {}
                item_data["itemId"] = row["itemId"]
                item_data["name"] = row["name"]
                item_data["width"] = int(row["width"])
                item_data["depth"] = int(row["depth"])
                item_data["height"] = int(row["height"])
                item_data["mass"] = float(row["mass"])
                item_data["priority"] = int(row["priority"])
                if row.get("expiryDate"):
                    item_data["expiryDate"] = datetime.strptime(row["expiryDate"], "%Y-%m-%d").date()
                item_data["usageLimit"] = int(row["usageLimit"])
                item_data["preferredZone"] = row["preferredZone"]

                item = ItemSchema(**item_data)
                db_item = Item(**item.dict())
                db.add(db_item)
                db.commit()
                items_imported += 1
            except (ValueError, KeyError) as e:
                errors.append({"row": row, "message": str(e)})

        return {
            "success": True,
            "itemsImported": items_imported,
            "errors": errors
        }
    except Exception as e:
        logging.error(f"Error importing items: {e}")
        raise


# API: Import Containers from CSV
@app.post("/api/import/containers", response_model=ImportResponse)
def import_containers(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = file.file.read().decode('utf-8')
        reader = csv.DictReader(contents.splitlines())
        containers_imported = 0
        errors = []

        for row in reader:
            try:
                container_data = {}
                container_data["containerId"] = row["containerId"]
                container_data["zone"] = row["zone"]
                container_data["width"] = int(row["width"])
                container_data["depth"] = int(row["depth"])
                container_data["height"] = int(row["height"])

                container = ContainerSchema(**container_data)
                db_container = Container(**container.dict())
                db.add(db_container)
                db.commit()
                containers_imported += 1
            except (ValueError, KeyError) as e:
                errors.append({"row": row, "message": str(e)})

        return {
            "success": True,
            "containersImported": containers_imported,
            "errors": errors
        }
    except Exception as e:
        logging.error(f"Error importing containers: {e}")
        raise


# API: Export Current Arrangement to CSV
@app.get("/api/export/arrangement")
def export_arrangement(db: Session = Depends(get_db)):
    try:
        items = db.query(Item).all()
        rows = []

        for item in items:
            rows.append([item.itemId, "contA", "(0,0,0)", f"({item.width},{item.depth},{item.height})"])  # Placeholder

        output = "\n".join([",".join(row) for row in rows])
        return output
    except Exception as e:
        logging.error(f"Error exporting arrangement: {e}")
        raise


# API: Get Logs
@app.get("/api/logs")
def get_logs(
    startDate: Optional[str] = Query(None, example="2025-03-10"),
    endDate: Optional[str] = Query(None, example="2025-03-15"),
    itemId: Optional[str] = Query(None, example="item001"),
    userId: Optional[str] = Query(None, example="astronaut1"),
    actionType: Optional[str] = Query(None, example="placement"),
    db: Session = Depends(get_db),
):
    try:
        logs = [
            {
                "timestamp": "2025-03-13T10:00:00",
                "userId": "astronaut1",
                "actionType": "placement",
                "itemId": "001",
                "details": {"fromContainer": "contA", "toContainer": "contB", "reason": "space optimization"},
            }
        ]  # Placeholder - Replace with actual logging logic
        return {"logs": logs}
    except Exception as e:
        logging.error(f"Error getting logs: {e}")
        raise


# ✅ Test Route
@app.get("/")
def home():
    return {"message": "Space Cargo API is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)