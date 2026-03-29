import { useState, useEffect } from "react"
import Header from "./components/Header"
import TextInput from "./components/TextInput"
import ResultCard from "./components/ResultCard"
import { analyzeText } from "./api"

export default function App() {
  const [text,     setText]     = useState("")
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [darkMode, setDarkMode] = useState(false)

  // Apply dark mode class to <html> element
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode)
  }, [darkMode])

  async function handleAnalyze() {
    if (!text.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await analyzeText(text)
      setResult(data)
    } catch (err) {
      setError("Could not connect to the API. Make sure your backend is running on port 8000.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">

      <Header darkMode={darkMode} setDarkMode={setDarkMode} />

      <main className="max-w-2xl mx-auto px-4 py-8 space-y-6">

        {/* Page title */}
        <div>
          <h1 className="text-2xl font-medium text-gray-900 dark:text-white">
            Bias Detection
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Paste any text to detect bias and see which words contribute to it
          </p>
        </div>

        {/* Input section */}
        <TextInput
          text={text}
          setText={setText}
          onAnalyze={handleAnalyze}
          loading={loading}
        />

        {/* Error message */}
        {error && (
          <div className="px-4 py-3 bg-red-50 dark:bg-red-900/30 border
                          border-red-200 dark:border-red-800 rounded-lg
                          text-sm text-red-700 dark:text-red-300">
            {error}
          </div>
        )}

        {/* Loading message */}
        {loading && (
          <div className="px-4 py-3 bg-blue-50 dark:bg-blue-900/30 border
                          border-blue-200 dark:border-blue-800 rounded-lg
                          text-sm text-blue-700 dark:text-blue-300">
            LIME is analyzing your text — this takes 30–60 seconds on CPU. Please wait...
          </div>
        )}

        {/* Results */}
        <ResultCard result={result} />

      </main>
    </div>
  )
}