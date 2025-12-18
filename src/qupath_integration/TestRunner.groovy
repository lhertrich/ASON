import static qupath.lib.gui.scripting.QPEx.*

// --- CONFIGURATION ---
// Change these to your actual paths!

def pythonExec = "/opt/anaconda3/envs/research_project/bin/python"  
def scriptPath = "/Users/levin/Documents/Uni/Master/semester_3/research_project/ASON/src/pipeline_cli/hello_bridge.py" 
def outputPath = buildFilePath(PROJECT_BASE_DIR, "bridge_test.json")

// --- EXECUTION ---
println "Starting bridge test..."

def command = [pythonExec, scriptPath, "--name", "QuPath_User", "--output", outputPath]

// This version captures the output so you can see Python's "print" statements
def process = new ProcessBuilder(command).redirectErrorStream(true).start()
process.inputStream.eachLine { println "PYTHON: " + it }
process.waitFor()

// --- CHECK RESULTS ---
if (new File(outputPath).exists()) {
    importObjectsFromFile(outputPath)
    println "Success! A square should have appeared at [0,0]."
} else {
    println "Error: Output file was not created. Check the paths above."
}