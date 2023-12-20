const { loadPyodide } = require("pyodide")
const fs = require("fs")

async function main() {
  let pyodide = await loadPyodide()
  await pyodide.loadPackage(["micropip"])

  output = await pyodide.runPythonAsync(`
    import micropip

    await micropip.install([
      "https://cdn.holoviz.org/panel/1.3.5/dist/wheels/bokeh-3.3.2-py3-none-any.whl",
      "https://cdn.holoviz.org/panel/1.3.5/dist/wheels/panel-1.3.5-py3-none-any.whl",
      "https://files.pythonhosted.org/packages/7f/0e/f3e09ad030185c5f2bb03265778699c8f80d7481b1e0abbb2a12c63fe093/holoviews-1.18.2a2-py2.py3-none-any.whl",
      "holonote",
      "hvplot",
    ])

    micropip.freeze()
    `)
  fs.writeFileSync("pyodide-lock.json", output)
}
main()
