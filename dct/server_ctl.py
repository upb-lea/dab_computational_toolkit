import base64
import io
import threading
import time

import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

class Dct_server:
    # Variable declaration
    # FastAPI-Server definieren
    app = FastAPI()
    templates = Jinja2Templates(directory="htmltemplates")
    status_message = "Warte auf Knopfdruck"  # Initialer Text
    # Serverobject
    srv_obj = None
    # Shared memory variable
    shared_histogram = None
    req_stop = None
    stop_flag = None

    # Mounten des Stylesheetpfades
    app.mount("/StyleSheets", StaticFiles(directory="htmltemplates/StyleSheets"), name="Stylesheets")

    @staticmethod
    def run_server(req_stop, stop_flag):
        """ Startet den FastAPI-Server """
        # Overtake the shared memory variable
        # Dct_server.shared_histogram = shared_histogram
        Dct_server.req_stop = req_stop
        Dct_server.stop_flag = stop_flag
        # Initialize server configuration
        config = uvicorn.Config(Dct_server.app, host="127.0.0.1", port=8004, log_level="info")
        Dct_server.srv_obj = uvicorn.Server(config)
        # Create thread for the server and start it
        Dct_server.srv_thread = threading.Thread(target=Dct_server.dct_server_thread)
        # Start the server (blocking call)
        Dct_server.srv_thread.start()
        # Supervice if the server is stopped by main
        while True:
            # Reduce CPU-supervice load by toggle each second
            time.sleep(1)
            # Check if server is requested to stop
            if Dct_server.stop_flag.value == 1 or Dct_server.req_stop.value == 1:
                Dct_server.stop_flag.value = 1
                break

        # Stoppt den Server
        Dct_server.srv_obj.should_exit = True
        # Wait for thread stop
        Dct_server.srv_thread.join()

    @staticmethod
    def dct_server_thread():
        """ Startet den FastAPI-Server """
        try:
            # Start the server in a blocking call
            Dct_server.srv_obj.run()
        except Exception as e:
            print(f"Fehler beim Starten des Servers: {e}")


    @app.get("/", response_class=HTMLResponse)
    async def main_page(request: Request, action: str = None):

        if action == "continue":
            Dct_server.status_message = "Weiter ist aktiv"
        elif action == "pause":
            Dct_server.status_message = "Pause ist aktiv"
        elif action == "stop":
            Dct_server.status_message = "Stoppt die Simulation und den Server"
            Dct_server.stop_flag.value = 1

        return Dct_server.templates.TemplateResponse("html_main.html", {"request": request, "textvariable": Dct_server.status_message})


    @app.get("/histogram")
    def get_histogram(request: Request):
        """Erstellt das Histogramm als Bild und gibt es als Base64-String zurück"""

        """
        plt.figure()
        plt.bar(range(25), Dct_server.shared_histogram[:25], color='blue', edgecolor='black')
        plt.xlabel("Schritte bis zur 5")
        plt.ylabel("Häufigkeit")
        plt.title("Histogramm der Wartezeiten bis zur 5")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        img_str = base64.b64encode(buf.read()).decode('utf-8')
        # Return the html-page with updated image data
        # return Dct_server.templates.TemplateResponse( "html_histogram.html", {"request": request, "imagedata": img_str})
        """
        imagepath = "StyleSheets/Dummytrafo.png"
        return Dct_server.templates.TemplateResponse( "html_histogram.html", {"request": request, "image_path": imagepath})
