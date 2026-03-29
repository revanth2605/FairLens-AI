const BASE_URL = "http://localhost:8000"

export async function analyzeText(text) {
    const response = await fetch(`${BASE_URL}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    })
    if (!response.ok) throw new Error("API request failed")
    return response.json()
}