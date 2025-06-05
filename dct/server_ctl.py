import multiprocessing
from multiprocessing import Queue
import threading
import time
import os
from os.path import abspath
from typing import Any
from enum import Enum

import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.sessions import SessionMiddleware

# own libraries
import server_ctl_dtos as srv_ctl_dtos

# Structure classes
# Structure class of request command
class ReqCmd(Enum):
    sd_main = 0 # Request statistical data of main page
    sd_detail = 1  # Request detailed statistical data
    pareto_front = 2 # Request pareto front

# Structure class of request
class SrvReqData:
    # Request command
    req_cmd: ReqCmd;
    # Id of circuit filtered point
    c_filt_pt_id: int

class DctServer:

    # Allocate FastAPI-Server
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

    #  Templates
    _template_directory = abspath("htmltemplates")

    templates = Jinja2Templates(directory=_template_directory)

    # Add password
    users = {"Andi": "hallo"}

    status_text = ["Idle", "InProgress", "Done","Skipped"]

    # Status icon definition to display
    icon = ["StyleSheets/OptIdle.png", "StyleSheets/OptInProgress.png", "StyleSheets/OptDone.png",
            "StyleSheets/OptSkipped.png"]

    # Allocate multi processing variables
    srv_request_queue: Queue
    srv_response_queue: Queue

    # ASA Create switch for release
    # Release:
    # _ssl_cert = os.getenv("SSL_CERT_PATH", "ssl/cert.pem")  # Default for development
    # _ssl_key = os.getenv("SSL_KEY_PATH", "ssl/key.pem")
    # Development
    _ssl_cert = abspath("ssl/cert.pem")  # Default für Entwicklung
    _ssl_key = abspath("ssl/key.pem")
    _optuna_path = abspath("htmltemplates/OptunaOrg.html")

    status_message = "Warte auf Knopfdruck"  # Initialer Text
    # Server object
    srv_obj = None
    # Shared memory variable
    req_stop = multiprocessing.Value('i', 0)
    stop_flag = multiprocessing.Value('i', 0)
    # Server process
    _server_process = None
    # program exit flag
    _prog_exit_flag = False
    # Server supervision
    _srv_supervision = None
    # Selected configuration id
    _c_config_id = 0
    # Selected filtered point id
    _c_filt_ptID = 0

    # Debug
    breakstatus: int = 0

    # Mount of stylesheet path
    app.mount("/StyleSheets", StaticFiles(directory=os.path.join(_template_directory, "StyleSheets")), name="Stylesheets")

    @staticmethod
    def get_current_user(request: Request):
        return request.session.get("user")

    @staticmethod
    def start_dct_server(act_srv_request_queue: Queue, act_srv_response_queue: Queue, program_exit_flag: bool):
        """Starts the server to control and supervice simulation.

        :param act_srv_request_queue: Queue object to request data from main process
        :type  act_srv_request_queue: Queue
        :param act_srv_response_queue: Queue object to responds to server process
        :type  act_srv_response_queue: Queue
        :param program_exit_flag: Flag, which indicates to terminate the program on request
        :type  program_exit_flag: boolean
        """

        DctServer._prog_exit_flag = program_exit_flag

        # Start the server process
        DctServer._server_process = multiprocessing.Process(target=DctServer._run_server,
                                                            args=(act_srv_request_queue, act_srv_response_queue,))
        DctServer._server_process.start()

        # Check if server process supervision is to start due to program exit requested by server
        if DctServer._prog_exit_flag:
            # Create thread for the serversupervision and start it
            DctServer._srv_supervision = threading.Thread(target=DctServer._supervice_server_stop, daemon=True)
            DctServer._srv_supervision.start()

    @staticmethod
    def stop_dct_server():
        """Stop the server for the control and supervisuib of the simulation.

        :param req_stop_server: Shared memory flag to request server to stop
        :type  req_stop_server: multiprocessing.Value
        """

        # Set program exit flag to false because program will be exit by themself
        DctServer._prog_exit_flag = False

        # Request server to stop
        DctServer.req_stop.value = 1
        # Debug
        print("Process shall join")
        # Wait for joined server process
        DctServer._server_process.join(5)
        # Stop server supervision if started
        if DctServer._srv_supervision != None:
            DctServer._srv_supervision.join(5)
        # Debug
        print("Process has joined")

    @staticmethod
    def _supervice_server_stop():
        """ Stops the supervision of the server in main process"""
        # Supervice if the server is stopped by user request
        while True:
            # Reduce CPU-supervice load by toggle each second
            time.sleep(1)
            # Check if server is stopped
            if DctServer.stop_flag.value == 1:
                # Requested to stop if the program needs to stop too
                if DctServer._prog_exit_flag:
                    # Check if the program needs to stop too
                    print("Program stop is requested")
                    # Hard kill of process
                    os._exit(0)
                break

    @staticmethod
    def _run_server(act_srv_request_queue: Queue, act_srv_response_queue: Queue):
        """ Start the FastAPI-Server."""
        # Overtake the shared memory variable
        DctServer.srv_request_queue = act_srv_request_queue
        DctServer.srv_response_queue = act_srv_response_queue
        # Start the server (blocking call)
        config = uvicorn.Config(DctServer.app, host="127.0.0.1", port=8007, log_level="info",
                                ssl_keyfile=DctServer._ssl_key, ssl_certfile=DctServer._ssl_cert)

        DctServer.srv_obj = uvicorn.Server(config)
        # Create thread for the server and start it
        DctServer.srv_thread = threading.Thread(target=DctServer.dct_server_thread, daemon=True)
        DctServer.srv_thread.start()
        # Supervice if the server is stopped by main
        while True:
            # Reduce CPU-supervice load by toggle each second
            time.sleep(1)
            # Check if server is requested to stop
            if DctServer.req_stop.value == 1:
                break

        # Stop the server
        DctServer.srv_obj.should_exit = True
        # Debug
        print("SThread soll joinen")
        # Wait for thread stop
        DctServer.srv_thread.join()
        # Debug
        print("SThread hat gejoint")
        # Set server stop flag to 0
        DctServer.stop_flag.value = 1

    @staticmethod
    def dct_server_thread():
        """ Start the FastAPI-server """
        # Start the server in a blocking call
        DctServer.srv_obj.run()

    @staticmethod
    def get_table_data(component_data_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]):
        # Variable declaration
        # Return value
        table_data = []
        # Index variable
        id = 0

        # Loop over the configurations
        for entry in component_data_list:
            table_data.append({"conf_name": entry.conf_name, "nbtrails": entry.nb_of_trials,
                               "nbfilt_pt": entry.progress_data.nb_of_filtered_points, "proc_time": DctServer.get_format_time(entry.progress_data.proc_run_time),
                               "imagelink": DctServer.icon[entry.progress_data.status], "status": DctServer.status_text[entry.progress_data.status],
                               "index": id})
            # Increment id
            id = id + 1
        return table_data

    @staticmethod
    def get_circuit_table_data(circuit_data: srv_ctl_dtos.CircuitConfigurationDataDto) -> dict:
        # Variable declaration
        # Enter the table data
        circuit_table_data:dict = {"conf_name": circuit_data.conf_name,
                                        "nbtrails": circuit_data.nb_of_trials,
                                        "name_of_filt_pt_list": circuit_data.name_of_filtered_points,
                                        "proc_time": DctServer.get_format_time(circuit_data.progress_data.proc_run_time),
                                        "imagelink": DctServer.icon[circuit_data.progress_data.status],
                                        "status": DctServer.status_text[circuit_data.progress_data.status], "index": 0}

        return circuit_table_data

    @staticmethod
    def get_heat_sink_table_data(act_heat_sink_data_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]) -> list[dict]:
        # Variable declaration
        # Return value
        heat_sink_table_data_list: list[dict[str, Any]] = []

        # Index variable
        entry_id = 0

        # Loop over the configurations
        for entry in act_heat_sink_data_list:
            # Enter the table data
            heat_sink_table_data_list.append({"conf_name": entry.conf_name, "nbtrails": entry.nb_of_trials,
                                              "proc_time": DctServer.get_format_time(entry.progress_data.proc_run_time),
                                              "imagelink": DctServer.icon[entry.progress_data.status],
                                              "status": DctServer.status_text[entry.progress_data.status], "index": entry_id})
            # Increment id
            entry_id = entry_id + 1

        return heat_sink_table_data_list

    @staticmethod
    def get_summary_table_data(act_summary_data_list: list[srv_ctl_dtos.SummaryDataEntryDto]) -> list[dict]:
        # Variable declaration
        # Return value
        summary_table_data_list: list[dict[str, Any]] = []

        # Index variable
        entry_id = 0

        # Loop over the configurations
        for entry in act_summary_data_list:
            # Enter the table data
            summary_table_data_list.append({"conf_name": entry.conf_name, "proc_time": DctServer.get_format_time(entry.progress_data.proc_run_time),
                                            "nb_of_combinations": entry.nb_of_combinations,
                                            "imagelink": DctServer.icon[entry.progress_data.status],
                                            "status": DctServer.status_text[entry.progress_data.status], "index": entry_id})
            # Increment id
            entry_id = entry_id + 1

        return summary_table_data_list

    @staticmethod
    def get_dummy_table_data(prefix_r: str):
        # Add content
        table_data = [
            {"conf_name": prefix_r + "_Config A", "nbtrails": 100, "nbfilt_pt": 10, "proc_time": "5s",
             "imagelink": "StyleSheets/OptDone.png", "status": "Done", "index": 1},
            {"conf_name": prefix_r + "_Config B", "nbtrails": 200, "nbfilt_pt": 20, "proc_time": "10s",
             "imagelink": "StyleSheets/OptInProgress.png", "status": "InProgress", "index": 2},
            {"conf_name": prefix_r + "_Config C", "nbtrails": 300, "nbfilt_pt": 30, "proc_time": "15s",
             "imagelink": "StyleSheets/OptIdle.png", "status": "Idle", "index": 3},
            {"conf_name": prefix_r + "_Config D", "nbtrails": 300, "nbfilt_pt": 30, "proc_time": "15s",
             "imagelink": "StyleSheets/OptSkipped.png", "status": "Skipped", "index": 4},
        ]
        return table_data

    @staticmethod
    def get_table_list_data(filt_point_num: int, offset: int, prefix_r: str):

        # Add content
        table_list_data = [
            {"conf_name": prefix_r + "_ConfA", "nbtrails": 100, "nbfilt_pt": offset + filt_point_num + 7,
             "proc_time": "5s",
             "imagelink": "StyleSheets/OptDone.png", "status": "Done", "index": 1},
            {"conf_name": prefix_r + "_ConfB", "nbtrails": 200, "nbfilt_pt": offset + filt_point_num + 10,
             "proc_time": "10s",
             "imagelink": "StyleSheets/OptInProgress.png", "status": "InProgress", "index": 2},
            {"conf_name": prefix_r + "_ConfC", "nbtrails": 300, "nbfilt_pt": offset + filt_point_num + 13,
             "proc_time": "15s",
             "imagelink": "StyleSheets/OptIdle.png", "status": "Idle", "index": 3}
        ]

        return table_list_data

    @staticmethod
    def get_filtered_point_list(nb_filt_pt: int):
        # ASA: To  replace by seeking the list on the drive or request list
        filterd_pt_list = []

        for i in range(nb_filt_pt):
            filterd_pt_list.append([str(i * 3 + 20) + "_pt", i])

        return filterd_pt_list

    @staticmethod
    def load_optuna_html_file(filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _create_info_string(buttonid: int, tableid: int, filt_pt_id: int, page_id: int) -> str:
        # Variable declaration
        info_string = ""

        # Check pageID
        if page_id == 0:
            if tableid == 1:
                info_string = f"Circuit configuration {buttonid}"
            elif tableid == 4:
                info_string = f"Heatsink configuration {buttonid}"
            elif tableid == 5:
                info_string = f"Summary of circuit configuration {buttonid}"
            else:
                info_string = "Unknown configuration "
        elif page_id == 1:
            if tableid == 1:
                info_string = f"Circuit configuration {DctServer._c_config_id}"
            elif tableid == 2:
                info_string = f"Inductor configuration {buttonid} of filtered circuit point {filt_pt_id}"
            elif tableid == 3:
                info_string = f"Transformer configuration {buttonid} of filtered circuit point {filt_pt_id}"
            elif tableid == 4:
                info_string = f"Heatsink configuration {buttonid}"
            elif tableid == 5:
                info_string = f"Summary of circuit configuration {buttonid}"
            else:
                info_string = "Unknown configuration "
        else:
            info_string = "Wrong html_id"

        return (info_string)

    @staticmethod
    def get_format_time(time_value: float)-> str:
        # 0-6000s
        if time_value<6000:
            format_time = "{:.0f}s".format(time_value)
        # 10min-100min
        elif time_value<60000:
            minutes = int(time_value // 60)
            seconds = int(time_value % 60)
            format_time = f"{minutes}min{seconds}s"
        # 1h40min- 99h59min
        elif time_value<360000:
            minutes = int(time_value // 60)
            hours = int(minutes // 60)
            minutes = int(minutes % 60)
            format_time = f"{hours}h{minutes}min"
        # >=100h
        else:
            hours = int(time_value // 3600)
            format_time = f"{hours}h"

        return format_time

    # -- Responses on client requests --------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    @app.get("/html_homepage1", response_class=HTMLResponse)
    async def main_page1(request: Request, action: str = None, user: str = Depends(get_current_user),
                         buttonid: int = None, tableid: int = None):
        if action == "logout":
            request.session.clear()
            user = None
            DctServer.status_message = "User ist ausgelogged"
        elif action == "paretocircuit":
            info_string = DctServer._create_info_string(buttonid, tableid, 0, 0)
            return DctServer.get_pareto_front(request, info_string, "/html_homepage1")
        elif action == "details" and tableid == 1:
            DctServer._c_config_id = buttonid
            return await DctServer.main_page2(request, None, 0, user)
        elif action == "controlsheet":
            # Check if user is authorized
            if not user == None:
                if DctServer.breakstatus == 1:
                    DctServer.breakstatus = 0
                else:
                    DctServer.breakstatus = 1

                breakstatus = DctServer.breakstatus
                return DctServer.templates.TemplateResponse("control_page.html",
                                                            {"request": request, "url_back": "/html_homepage1",
                                                             "breakstatus": breakstatus})

        # Init request for main process
        request_data=SrvReqData()
        request_data.req_cmd=ReqCmd.sd_main

        # Request data from main process
        DctServer.srv_request_queue.put(request_data)
        # Wait for response
        data: srv_ctl_dtos.QueueMainData = DctServer.srv_response_queue.get()

        # Add content circuit config
        # Create list (in future it is a list of configurations)
        circuit_conf_list: list[srv_ctl_dtos.ConfigurationDataEntryDto] = data.circuit_list
        table_data_circuit = DctServer.get_table_data(circuit_conf_list)

        # Add content heatsink config
        # table_data_heatsink = DctServer.get_dummy_table_data("Ht")
        table_data_heatsink = DctServer.get_heat_sink_table_data(data.heat_sink_list)

        # Add content summary
        # table_data_summary = DctServer.get_dummy_table_data("s")
        table_data_summary = DctServer.get_summary_table_data(data.summary_list)

        return DctServer.templates.TemplateResponse("main_page1.html",
                                                    {"request": request, "ctable_data": table_data_circuit,
                                                     "htable_data": table_data_heatsink,
                                                     "stable_data": table_data_summary,
                                                     "total_proc_time": DctServer.get_format_time(data.total_proc_time),
                                                     "total_inductor_proc_time": DctServer.get_format_time(data.inductor_proc_time),
                                                     "total_transformer_proc_time": DctServer.get_format_time(data.transformer_proc_time),
                                                     "textvariable": DctServer.status_message,
                                                     "user": user})

    @app.get("/html_homepage2", response_class=HTMLResponse)
    async def main_page2(request: Request, action: str = None, c_filt_ptID: int = 0,
                         user: str = Depends(get_current_user),
                         buttonid: int = None, tableid: int = None):
        # Check if an action is requested
        # User request: Logout
        if action == "logout":
            request.session.clear()
            user = None
            DctServer.status_message = "User is logged out"
        # User request: Button press to display Pareto-front
        elif action == "paretocircuit":
            info_string = DctServer._create_info_string(buttonid, tableid, DctServer._c_filt_ptID, 1)
            # Display the Pareto-front
            return DctServer.get_pareto_front(request, info_string, "/html_homepage2")
        # User request: Button press to change to control sheet
        elif action == "controlsheet":
            # Check if user is authorized
            if not user == None:
                if DctServer.breakstatus == 1:
                    DctServer.breakstatus = 0
                else:
                    DctServer.breakstatus = 1
                # Display the control page
                breakstatus = DctServer.breakstatus
                return DctServer.templates.TemplateResponse("control_page.html",
                                                            {"request": request, "url_back": "/html_homepage2",
                                                             "breakstatus": breakstatus})

        # Init request for main process
        request_data=SrvReqData()
        request_data.req_cmd=ReqCmd.sd_detail
        request_data.c_filt_pt_id = c_filt_ptID

        # Save the selected filtered point id
        DctServer._c_filt_ptID=c_filt_ptID

        # Request data from main process
        DctServer.srv_request_queue.put(request_data)
        # Wait for response
        data: srv_ctl_dtos.QueueDetailData = DctServer.srv_response_queue.get()

        # Add content circuit config
        table_data_circuit = DctServer.get_circuit_table_data(data.circuit_data)

        # Add content inductor config
        table_data_inductor = DctServer.get_table_data(data.inductor_list)

        # Add content transformer config
        table_data_transformer = DctServer.get_table_data(data.transformer_list)

        # Add content heatsink config
        # table_data_heatsink = DctServer.get_dummy_table_data("Ht")
        table_data_heatsink = DctServer.get_heat_sink_table_data(data.heat_sink_list)

        # Add content summary
        # table_data_summary = DctServer.get_dummy_table_data("s")
        table_data_summary = DctServer.get_summary_table_data([data.summary_data])

        return DctServer.templates.TemplateResponse("main_page2.html", {"request": request,
                                                                        "ctable_data": table_data_circuit,
                                                                        "itable_data": table_data_inductor,
                                                                        "ttable_data": table_data_transformer,
                                                                        "htable_data": table_data_heatsink,
                                                                        "stable_data": table_data_summary,
                                                                        "conf_proc_time": DctServer.get_format_time(data.conf_proc_time),
                                                                        "c_filt_ptID": c_filt_ptID,
                                                                        "textvariable": DctServer.status_message,
                                                                        "user": user})

    @app.get("/control_page", response_class=HTMLResponse)
    async def control_page(request: Request, action: str = None, url_back: str = "/html_homepage1"):
        if action == "continue":
            DctServer.status_message = "Continue is active"
        elif action == "pause":
            DctServer.status_message = "Pause is active"
        elif action == "stop":
            DctServer.status_message = "Stops the server and the optmisation (if prog_exit_flag==true)"
            DctServer.req_stop.value = 1

        return DctServer.templates.TemplateResponse("control_page.html",
                                                    {"request": request, "url_back": url_back,
                                                     "breakstatus": DctServer.breakstatus})

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return DctServer.templates.TemplateResponse("login.html", {"request": request})

    @app.post("/login")
    async def login(request: Request, username: str = Form(...), password: str = Form(...)):
        if DctServer.users.get(username) == password:
            request.session["user"] = username
            return RedirectResponse(url="/", status_code=303)
        return DctServer.templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page(request: Request, user: str = Depends(get_current_user)):
        if not user:
            return RedirectResponse(url="/login")
        return DctServer.templates.TemplateResponse("admin.html", {"request": request, "user": user})

    @app.get("/paretofront")
    def get_pareto_front(request: Request, info_string_r: str = "", url_back: str = None):
        """Provide the Pareto-front"""
        # Later to replace by request of optuna paretofront data
        original_html = DctServer.load_optuna_html_file(DctServer._optuna_path)

        # Add information and back-button
        insert_html = "<h1>Actual Paretofront</h1>" + "<p>" + info_string_r + "</p>"
        insert_html = insert_html + "<button onclick=\"location.href='" + url_back
        insert_html = insert_html + "'\">Back to main menue</button> "

        # Search for tag body
        body_index = original_html.lower().find("<body>")
        if body_index == -1:
            return HTMLResponse(content="Fehler: <body> Tag nicht gefunden", status_code=500)

        # Seek position to add
        insertion_point = body_index + len("<body>")
        # Add values
        modified_html = (
            original_html[:insertion_point] +
            "\n" + insert_html + "\n" +
            original_html[insertion_point:]
        )

        return HTMLResponse(content=modified_html)

        # Return the html-page with updated image data
        # return Dct_server.templates.TemplateResponse( "pareto_dummy.html", {"request": request, "imagedata": img_str})
        # imagepath = "StyleSheets/Dummytrafo.png"
        # return Dct_server.templates.TemplateResponse( "pareto_dummy.html", {"request": request, "image_path": imagepath, "act_button_id": act_button_id, "act_table_id": act_table_id, "url_back": url_back})


