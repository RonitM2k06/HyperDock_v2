const API_URL = "http://127.0.0.1:8000"; // FastAPI backend URL

// Function to add a new storage container
async function addContainer() {
    const containerName = document.getElementById("containerName").value;
    
    if (!containerName) {
        alert("Please enter a container name");
        return;
    }

    const response = await fetch(`${API_URL}/add-container`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: containerName })
    });

    if (response.ok) {
        alert("Container added successfully!");
        document.getElementById("containerName").value = ""; // Clear input
        getContainers(); // Reload container list
    } else {
        alert("Failed to add container");
    }
}

// Function to get and display containers
async function getContainers() {
    const response = await fetch(`${API_URL}/get-containers`);
    const containers = await response.json();
    
    const containerList = document.getElementById("containerList");
    containerList.innerHTML = ""; // Clear list

    containers.forEach(container => {
        const li = document.createElement("li");
        li.textContent = `Container ID: ${container.id}, Name: ${container.name}`;
        containerList.appendChild(li);
    });
}
