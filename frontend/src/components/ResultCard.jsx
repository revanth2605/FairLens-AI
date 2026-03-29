export default function ResultCard({ result }) {
  if (!result) return null

  const isBiased   = result.label === "Biased"
  const confidence = Math.round(result.confidence * 100)

  // Build a map of word → score for fast lookup
  const wordScores = {}
  result.explanation.forEach(item => {
    wordScores[item.word.toLowerCase()] = item.score
  })

  // Highlight function — colors each word based on its LIME score
  function getWordStyle(word) {
    const clean = word.toLowerCase().replace(/[^a-z]/g, "")
    const score = wordScores[clean]

    if (!score || Math.abs(score) < 0.01) return {}

    if (score > 0) {
      // Biased word → red background, intensity based on score
      const alpha = Math.min(Math.abs(score) * 3, 0.7).toFixed(2)
      return { background: `rgba(220, 50, 50, ${alpha})`,
               borderRadius: "3px", padding: "1px 4px" }
    } else {
      // Counter-bias word → green background
      const alpha = Math.min(Math.abs(score) * 3, 0.7).toFixed(2)
      return { background: `rgba(34, 197, 94, ${alpha})`,
               borderRadius: "3px", padding: "1px 4px" }
    }
  }

  // Split original text into tokens and highlight each one
  const tokens = result.text.split(" ")

  return (
    <div className={`rounded-xl border overflow-hidden transition-all
                     ${isBiased
                       ? "border-red-200 dark:border-red-800"
                       : "border-green-200 dark:border-green-800"}`}>

      {/* Top banner — label + confidence */}
      <div className={`px-6 py-4 flex items-center justify-between
                       ${isBiased
                         ? "bg-red-50 dark:bg-red-900/30"
                         : "bg-green-50 dark:bg-green-900/30"}`}>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Result</p>
          <p className={`text-2xl font-medium
                         ${isBiased
                           ? "text-red-600 dark:text-red-400"
                           : "text-green-600 dark:text-green-400"}`}>
            {result.label}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Confidence</p>
          <p className={`text-3xl font-medium
                         ${isBiased
                           ? "text-red-600 dark:text-red-400"
                           : "text-green-600 dark:text-green-400"}`}>
            {confidence}%
          </p>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 space-y-5">

        {/* Confidence bar */}
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
            Confidence breakdown
          </p>
          <div className="flex gap-3">
            <div className="flex-1 bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-400 mb-1">Not Biased</p>
              <p className="text-lg font-medium text-green-600 dark:text-green-400">
                {Math.round(result.scores["Not Biased"] * 100)}%
              </p>
            </div>
            <div className="flex-1 bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-400 mb-1">Biased</p>
              <p className="text-lg font-medium text-red-600 dark:text-red-400">
                {Math.round(result.scores["Biased"] * 100)}%
              </p>
            </div>
          </div>

          {/* Visual progress bar */}
          <div className="mt-3 h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-red-500 rounded-full transition-all duration-500"
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>

        {/* Word highlights */}
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
            Word importance
            <span className="ml-2 text-gray-400">
              (red = biased, green = counter-bias)
            </span>
          </p>
          <div className="text-sm leading-8 text-gray-800 dark:text-gray-200">
            {tokens.map((token, i) => (
              <span key={i} style={getWordStyle(token)}>
                {token}{" "}
              </span>
            ))}
          </div>
        </div>

        {/* Top biased words list */}
        {result.explanation.filter(w => w.direction === "biased").length > 0 && (
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
              Top biased words
            </p>
            <div className="flex flex-wrap gap-2">
              {result.explanation
                .filter(w => w.direction === "biased")
                .slice(0, 5)
                .map((item, i) => (
                  <span key={i}
                        className="px-2 py-1 bg-red-50 dark:bg-red-900/30
                                   text-red-700 dark:text-red-300
                                   text-xs rounded-md border border-red-100
                                   dark:border-red-800">
                    {item.word}
                    <span className="ml-1 opacity-60">
                      {(item.score * 100).toFixed(0)}%
                    </span>
                  </span>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}