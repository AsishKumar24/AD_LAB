function analyzeSentiment() {
    const text = document.getElementById("textInput").value;
    if (!text) {
        alert("Please enter text!");
        return;
    }

    fetch(`/sentiment?text=${encodeURIComponent(text)}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerText = `Sentiment: ${data.sentiment}`;
        })
        .catch(error => console.error("Error:", error));
}
