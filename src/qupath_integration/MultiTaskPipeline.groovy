import qupath.lib.scripting.QP
import qupath.fx.dialogs.Dialogs
import qupath.lib.regions.RegionRequest

def task = Dialogs.showChoiceDialog("Select Task", "Choose a pipeline step", ["tissue", "nuclei"], "nuclei")
if (task == null) return

def probInput = Dialogs.showInputDialog("Probability", "Enter StarDist threshold", 0.25)
if (probInput == null) return
double prob = probInput.doubleValue()

def pythonExec = "/opt/anaconda3/envs/research_project/bin/python"
def scriptPath = "/Users/levin/Documents/Uni/Master/semester_3/research_project/ASON/src/pipeline_cli/bridge.py"
def imgPath = QP.buildFilePath(QP.PROJECT_BASE_DIR, "tile_in.tif")
def jsonPath = QP.buildFilePath(QP.PROJECT_BASE_DIR, "results_out.json")

println "Exporting image..."
def server = QP.getCurrentServer()

double downsample = 1.0
int x = 0
int y = 0
int width = server.getWidth()
int height = server.getHeight()

def request = RegionRequest.createInstance(server.getPath(), downsample, x, y, width, height)
QP.writeImageRegion(server, request, imgPath)

println "Running Python: " + task
def command = [
    pythonExec, scriptPath, 
    "--image_path", imgPath, 
    "--output_json", jsonPath, 
    "--task", task, 
    "--prob_thresh", prob.toString()
]

def proc = new ProcessBuilder(command).redirectErrorStream(true).start()
proc.inputStream.eachLine { println "PYTHON: " + it }
proc.waitFor()

if (new File(jsonPath).exists()) {
    QP.importObjectsFromFile(jsonPath)
    QP.fireHierarchyUpdate()
    
    def viewer = getCurrentViewer()
    viewer.setAnnotationVisibility(true)
    viewer.setDetectionVisibility(true)
    getCurrentViewer().setVisualStyle(qupath.lib.gui.viewer.QuPathViewer.VisualStyle.ANNOTATIONS_SHOW, true)
    getCurrentViewer().setVisualStyle(qupath.lib.gui.viewer.QuPathViewer.VisualStyle.DETECTIONS_SHOW, true)
    println "Finished successfully!"
} else {
    println "Error: Result file not found. Check the console log above."
}