"""Server class implementation."""
# python libraries
import multiprocessing
import threading
import time

# 3rd party libraries
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import optuna
from optuna.visualization import plot_pareto_front

# own libraries


# Debug server
import logging
logging.basicConfig(level=logging.DEBUG)

class Dct_server:
    """server class to supervise the simulation."""

    # Variable declaration
    # FastAPI-Server definieren
    app = FastAPI()
    templates = Jinja2Templates(directory="/home/andreas/Workspace/Projekt/dab_computational_toolkit/dct/htmltemplates")
    status_message = "Warte auf Knopfdruck"  # Initialer Text
    # Serverobject
    srv_obj = None
    # Shared memory variable
    req_stop = multiprocessing.Value('i', 0)
    stop_flag = multiprocessing.Value('i', 0)
    # Server process
    _server_process = None
    # program exit flag
    _prog_exit_flag = False
    # Server supervision thread
    _srv_supervision_thd = None

    # Mounten des Stylesheetpfades
    app.mount("/StyleSheets", StaticFiles(directory="htmltemplates/StyleSheets"), name="Stylesheets")

    @staticmethod
    def start_dct_server(shared_histogram, program_exit_flag: bool):
        """Start the server to control and supervise simulation.

        :param shared_histogram: Shared memory flag for histogram information
        :type  shared_histogram: multiprocessing.Value
        :param program_exit_flag: Flag, which indicates if the server (False) or the whole simulation (True) is to stop
        :type  program_exit_flag: boolean
        """
        Dct_server._prog_exit_flag = program_exit_flag

        # Start the server process
        Dct_server._server_process = multiprocessing.Process(target=Dct_server._run_server, args=(shared_histogram,))
        Dct_server._server_process.start()
        # Check if server process supervision is to start due to program exit requested by server
        if Dct_server._prog_exit_flag:
            # Create thread for the serversupervision and start it
            Dct_server._srv_supervision_thd = threading.Thread(target=Dct_server._supervice_server_stop)
            Dct_server._srv_supervision_thd.start()

    @staticmethod
    def stop_dct_server():
        """Stop the simulation supervision server."""
        # Set program exit flag to false because program will be exit by themself
        Dct_server._prog_exit_flag = False

        # Request server to stop
        Dct_server.req_stop.value = 1
        # Debug
        print("Process shall join")
        # Wait for joined server process
        Dct_server._server_process.join(5)
        # Stop server supervision if started
        if Dct_server._srv_supervision_thd is not None:
            Dct_server._srv_supervision_thd.join(5)
        # Debug
        print("Process has joined")

    @staticmethod
    def _supervice_server_stop():
        """Stop the FastAPI-Server."""
        # Supervice if the server is stopped by user request
        while True:
            # Reduce CPU-supervice load by toggle each second
            time.sleep(1)
            # Check if server is stopped and requested to stop if the program needs to stop too
            if Dct_server.stop_flag.value == 1 and Dct_server._prog_exit_flag:
                # Check if the program needs to stop too
                print("Program stop is requested")
                # Soft kill of process does not work
                # sys.exit()
                break

    @staticmethod
    def _run_server(shared_histogram):
        """Start FastAPI-server.

        :param request : Request value
        :type  request : Request

        :return: Html- page based on html-template
        :rtype: _TemplateResponse
        """
        # Overtake the shared memory variable
        Dct_server.shared_histogram = shared_histogram
        # Start the server (blocking call)
        config = uvicorn.Config(Dct_server.app, host="127.0.0.1", port=8005, log_level="info")
        Dct_server.srv_obj = uvicorn.Server(config)
        # Create thread for the server and start it
        Dct_server.srv_thread = threading.Thread(target=Dct_server.dct_server_thread)
        Dct_server.srv_thread.start()
        # Supervice if the server is stopped by main
        while True:
            # Reduce CPU-supervice load by toggle each second
            time.sleep(1)
            # Check if server is requested to stop
            if Dct_server.req_stop.value == 1:
                break

        # Stoppt den Server
        Dct_server.srv_obj.should_exit = True
        # Debug
        print("SThread soll joinen")
        # Wait for thread stop
        Dct_server.srv_thread.join()
        # Debug
        print("SThread hat gejoint")
        # Set server stop flag to 0
        Dct_server.stop_flag.value = 1

    @staticmethod
    def dct_server_thread():
        """Start FastAPI-Server in thread."""
        # Start the server in a blocking call
        Dct_server.srv_obj.run()

    @staticmethod
    def LoadActualParetofront():
        # Verbinde dich mit der bestehenden Optuna-Datenbank
        study = optuna.load_study(study_name="circuit_01", storage="sqlite:////home/andreas/Workspace/Projekt/dab_computational_toolkit/workspace/2025-01-31_example/01_circuit/circuit_01/circuit_01.sqlite3")

        # Erzeuge die aktuelle Paretofront
        fig = plot_pareto_front(study)

        # Speichere die HTML-Darstellung des Plots in einer Variablen
        html_variable = fig.to_html(full_html=False)

        return html_variable


    @app.get("/", response_class=HTMLResponse)
    async def main_page(request: Request, action: str = None):
        """Provide the answer on client requests.

        :param request : Request value
        :type  request : Request
        :param action : Requested action
        :type  action : Requested action

        :return: Html- page based on html-template
        :rtype: _TemplateResponse
        """
        if action == "continue":
            Dct_server.status_message = "Weiter ist aktiv"
        elif action == "pause":
            Dct_server.status_message = "Pause ist aktiv"
        elif action == "stop":
            Dct_server.status_message = "Stoppt den Server und die Simulation (wenn prog_exit_flag==true)"
            Dct_server.req_stop.value = 1

        return Dct_server.templates.TemplateResponse("html_main.html",{"request": request, "textvariable": Dct_server.status_message})


    @app.get("/histogram", response_class=HTMLResponse)
    def get_histogram(request: Request):
        """Provide the answer on client histogram request.

        :param request : Request value
        :type  request : Request

        :return: Html- page based on html-template with Histogram information
        :rtype: _TemplateResponse
        """

        # Return the html-page with updated image data
        html_page = Dct_server.LoadActualParetofront()
        return HTMLResponse(content=html_page)

        # Return the html-page with updated image data
        # return Dct_server.templates.TemplateResponse( "html_histogram.html", {"request": request, "imagedata": img_str})
        # imagepath = "StyleSheets/Dummytrafo.png"
        # return Dct_server.templates.TemplateResponse("html_histogram.html", {"request": request, "image_path": imagepath})
