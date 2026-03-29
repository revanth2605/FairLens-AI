export default function Header({ darkMode, setDarkMode }) {
  return (
    <header className="border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-red-100 dark:bg-red-900 flex items-center justify-center">
          <span className="text-red-600 dark:text-red-300 text-sm font-bold">F</span>
        </div>
        <span className="text-lg font-medium text-gray-900 dark:text-white">FairLens AI</span>
        <span className="text-xs text-gray-400 dark:text-gray-500">Bias Detection</span>
      </div>

      {/* Dark mode toggle */}
      <button
        onClick={() => setDarkMode(!darkMode)}
        className="text-sm px-3 py-1.5 rounded-lg border border-gray-200
                   dark:border-gray-600 text-gray-600 dark:text-gray-300
                   hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
      >
        {darkMode ? "Light mode" : "Dark mode"}
      </button>
    </header>
  )
}