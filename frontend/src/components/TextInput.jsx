export default function TextInput({ text, setText, onAnalyze, loading }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200
                    dark:border-gray-700 p-6">

      <label className="block text-sm text-gray-500 dark:text-gray-400 mb-2">
        Enter text to analyze
      </label>

      {/* Text area */}
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type or paste any text here... e.g. Women are bad at driving."
        rows={4}
        className="w-full px-4 py-3 rounded-lg border border-gray-200 dark:border-gray-600
                   bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white
                   text-sm resize-none focus:outline-none focus:ring-2
                   focus:ring-blue-500 placeholder-gray-400 dark:placeholder-gray-500
                   transition-colors"
      />

      {/* Analyze button */}
      <button
        onClick={onAnalyze}
        disabled={loading || !text.trim()}
        className="mt-4 w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700
                   disabled:bg-gray-300 dark:disabled:bg-gray-600
                   text-white font-medium text-sm transition-colors
                   flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            {/* Spinner */}
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10"
                      stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor"
                    d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            Analyzing... (30-60 sec on CPU)
          </>
        ) : (
          "Analyze for Bias"
        )}
      </button>
    </div>
  )
}