"""Server control class to visualize current metrics."""
import multiprocessing
from multiprocessing import Queue
import threading
import time
import os
from os.path import abspath
from typing import Any
from enum import Enum

import uvicorn
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import _TemplateResponse
from starlette.middleware.sessions import SessionMiddleware

# own libraries
import dct.server_ctl_dtos as srv_ctl_dtos

# Structure classes
# Structure class of request command
class ReqCmd(Enum):
    """Enum of possible commands."""

    sd_main = 0       # Request statistical data of main page
    sd_detail = 1     # Request detailed statistical data
    pareto_front = 2  # Request pareto front

# Structure class of request
class SrvReqData:
    """Request command structure."""

    # Request command
    req_cmd: ReqCmd
    # Id of circuit filtered point
    c_filt_pt_id: int

class DctServer:
    """Server to visualize the actual progress and calculated Pareto-fronts."""

    # Methode deklaration
    srv_thread: threading.Thread

    # Allocate FastAPI-Server
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

    # Get path of this file
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    # Set template directory
    _template_directory = os.path.join(_base_dir, "htmltemplates")

    templates = Jinja2Templates(directory=_template_directory)

    # Currently dummy password, later to secure in specific way
    users = {"Andi": "hallo"}

    status_text = ["Idle", "InProgress", "Done", "Skipped"]

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

    status_message = "Wait for button press!"  # Initial text
    # Server object
    srv_obj: uvicorn.Server
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
    _c_filt_pt_ID = 0

    # Debug
    break_status: int = 0

    # Mount of stylesheet path
    app.mount("/StyleSheets", StaticFiles(directory=os.path.join(_template_directory, "StyleSheets")), name="Stylesheets")

    @staticmethod
    def get_current_user(request: Request) -> Any:
        """Provide the user of the current session.

        User of the current session in case of valid login

        :param request: Request information of the client request
        :type  request: Request
        :return: User name, if client is logged in, otherwise None
        :rtype: Any
        """
        retval = request.session.get("user")
        # return request.session.get("user")
        return retval

    @staticmethod
    def start_dct_server(act_srv_request_queue: Queue, act_srv_response_queue: Queue, program_exit_flag: bool) -> None:
        """Start the server to control and supervise simulation.

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
            DctServer._srv_supervision = threading.Thread(target=DctServer._supervise_server_stop, daemon=True)
            DctServer._srv_supervision.start()

    @staticmethod
    def stop_dct_server() -> None:
        """Stop the server for the control and supervision of the simulation."""
        # Set program exit flag to false because program will be exit by themself
        DctServer._prog_exit_flag = False

        # Request server to stop
        DctServer.req_stop.value = 1
        # Stop the server  process (if started)
        if DctServer._server_process is not None:
            DctServer._server_process.join(5)
        # Stop server supervision if started
        if DctServer._srv_supervision is not None:
            DctServer._srv_supervision.join(5)

    @staticmethod
    def _supervise_server_stop() -> None:
        """Stop the supervision of the server in main process."""
        # Supervice if the server is stopped by user request
        while True:
            # Reduce CPU-supervise load by toggle each second
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
    def _run_server(act_srv_request_queue: Queue, act_srv_response_queue: Queue) -> None:
        """Start the FastAPI-Server.

        :param act_srv_request_queue: Queue object to request data from main process
        :type  act_srv_request_queue: Queue
        :param act_srv_response_queue: Queue object to responds to server process
        :type  act_srv_response_queue: Queue
        """
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
            # Reduce CPU-supervise load by toggle each second
            time.sleep(1)
            # Check if server is requested to stop
            if DctServer.req_stop.value == 1:
                break

        # Stop the server
        DctServer.srv_obj.should_exit = True
        # Wait for thread stop
        DctServer.srv_thread.join()
        # Set server stop flag to 0
        DctServer.stop_flag.value = 1

    @staticmethod
    def dct_server_thread() -> None:
        """Start the FastAPI-server."""
        # Start the server in a blocking call
        DctServer.srv_obj.run()

    @staticmethod
    def get_table_data(component_data_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]) -> list[dict]:
        """Fill the table data to display progress based on the configuration progress data.

        :param component_data_list: List of configuration data for progress reporting
        :type  component_data_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]
        :return: List of dict with the formatted entries
        :rtype:  list[dict]
        """
        # Variable declaration
        # Return value
        table_data: list[dict] = []
        # Index variable
        id = 0

        # Loop over the configurations
        for entry in component_data_list:
            table_data.append({"conf_name": entry.conf_name, "nb_trails": entry.nb_of_trials,
                               "nb_filt_pts": entry.progress_data.nb_of_filtered_points,
                               "proc_time": DctServer.get_format_time(entry.progress_data.proc_run_time),
                               "image_link": DctServer.icon[entry.progress_data.status],
                               "status": DctServer.status_text[entry.progress_data.status],
                               "index": id})
            # Increment id
            id = id + 1
        return table_data

    @staticmethod
    def get_circuit_table_data(circuit_data: srv_ctl_dtos.CircuitConfigurationDataDto) -> dict:
        """Fill the table data for display circuit progress data of one configuration with filtered point name.

        :param circuit_data: Configuration data of circuit configuration for progress reporting
        :type  circuit_data: srv_ctl_dtos.CircuitConfigurationDataDto
        :return: Formatted entries of circuit configuration data for progress reporting
        :rtype:  dict
        """
        # Variable declaration
        # Enter the table data
        circuit_table_data: dict = {"conf_name": circuit_data.conf_name,
                                    "nb_trails": circuit_data.nb_of_trials,
                                    "filt_pts_name_list": circuit_data.filtered_points_name_list,
                                    "proc_time": DctServer.get_format_time(circuit_data.progress_data.proc_run_time),
                                    "image_link": DctServer.icon[circuit_data.progress_data.status],
                                    "status": DctServer.status_text[circuit_data.progress_data.status], "index": 0}

        return circuit_table_data

    @staticmethod
    def get_heat_sink_table_data(act_heat_sink_data_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]) -> list[dict]:
        """Fill the table data to display progress based on the configuration progress data.

        :param act_heat_sink_data_list: List of configuration data for progress reporting
        :type  act_heat_sink_data_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]
        :return: List of dict with the formatted entries
        :rtype:  list[dict]
        """
        # Variable declaration
        # Return value
        heat_sink_table_data_list: list[dict[str, Any]] = []

        # Index variable
        entry_id = 0

        # Loop over the configurations
        for entry in act_heat_sink_data_list:
            # Enter the table data
            heat_sink_table_data_list.append({"conf_name": entry.conf_name, "nb_trails": entry.nb_of_trials,
                                              "proc_time": DctServer.get_format_time(entry.progress_data.proc_run_time),
                                              "image_link": DctServer.icon[entry.progress_data.status],
                                              "status": DctServer.status_text[entry.progress_data.status], "index": entry_id})
            # Increment id
            entry_id = entry_id + 1

        return heat_sink_table_data_list

    @staticmethod
    def get_summary_table_data(act_summary_data_list: list[srv_ctl_dtos.SummaryDataEntryDto]) -> list[dict]:
        """Fill the table data to display summary progress based on the configuration progress data.

        :param act_summary_data_list: List of configuration data for progress reporting
        :type  act_summary_data_list: list[srv_ctl_dtos.SummaryDataEntryDto]
        :return: List of dict with the formatted entries
        :rtype:  list[dict]
        """
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
                                            "image_link": DctServer.icon[entry.progress_data.status],
                                            "status": DctServer.status_text[entry.progress_data.status], "index": entry_id})
            # Increment id
            entry_id = entry_id + 1

        return summary_table_data_list

    @staticmethod
    def load_optuna_html_file(filepath: str) -> str:
        """Load the optuna file to string.

        !!!Later to replace
        :param filepath: File name inclusive path
        :type  filepath: str
        :return: file data as string
        :rtype:  str
        """
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _create_info_string(button_id: int = -1, table_id: int = -1, filt_pt_id: int = -1, page_id: int = -1) -> str:
        """Create the information string based on the input parameters.

        With the input parameter the information string is generated.
        :param button_id: Index of the button, which is pressed
        :type  button_id: int
        :param table_id: Index of the table, which contains the button
        :type  table_id: int
        :param filt_pt_id: Index of actual filter point id
        :type  filt_pt_id: int
        :param page_id: Actual html-page
        :type  page_id: int
        :return: Information string
        :rtype:  str
        """
        # Variable declaration
        info_string = ""

        # Check pageID
        if page_id == 0:
            if table_id == 1:
                info_string = f"Circuit configuration {button_id}"
            elif table_id == 4:
                info_string = f"Heat sink configuration {button_id}"
            elif table_id == 5:
                info_string = f"Summary of circuit configuration {button_id}"
            else:
                info_string = "Unknown configuration "
        elif page_id == 1:
            if table_id == 1:
                info_string = f"Circuit configuration {DctServer._c_config_id}"
            elif table_id == 2:
                info_string = f"Inductor configuration {button_id} of filtered circuit point {filt_pt_id}"
            elif table_id == 3:
                info_string = f"Transformer configuration {button_id} of filtered circuit point {filt_pt_id}"
            elif table_id == 4:
                info_string = f"Heat sink configuration {button_id}"
            elif table_id == 5:
                info_string = f"Summary of circuit configuration {button_id}"
            else:
                info_string = "Unknown configuration "
        else:
            info_string = "Wrong html_id"

        return (info_string)

    @staticmethod
    def get_format_time(time_value: float) -> str:
        """Create the information string based on the input parameters.

        The displayed time value is displayed in human understandable way.
        Depending on thresholds the time is displayed in different way.
        With the input parameter the information string is generated.
        :param time_value: time value in second
        :type  time_value: float
        :return: formatted time value to match the display area
        :rtype:  str
        """
        # 0-6000s
        if time_value < 6000:
            format_time = "{:.0f}s".format(time_value)
        # 10min-100min
        elif time_value < 60000:
            minutes = int(time_value // 60)
            seconds = int(time_value % 60)
            format_time = f"{minutes}min{seconds}s"
        # 1h40min- 99h59min
        elif time_value < 360000:
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

    @staticmethod
    @app.get("/", response_class=HTMLResponse, response_model=None)
    @app.get("/html_homepage1", response_class=HTMLResponse, response_model=None)
    async def main_page1(request: Request, action: str = "", user: str = Depends(get_current_user),
                         button_id: int | None = None, table_id: int | None = None) -> _TemplateResponse | HTMLResponse:
        """Provide the html-information based on client request to html_homepage1.

        :param request: Request information of the client request
        :type  request: Request
        :param action: Information about the requested action (Keyword driven)
        :type  action: str
        :param user: User, in case of valid user
        :type  user: str
        :param button_id: Index of selected button, if a button was pressed
        :type  button_id: int
        :param table_id: Index of selected table in case of a button press
        :type  table_id: int
        :return: html-page
        :rtype:  _TemplateResponse | HTMLResponse
        """
        if action == "logout":
            request.session.clear()
            user = ""
            DctServer.status_message = "User is logged off"
        elif action == "pareto_circuit":
            # Check for valid parameters
            if button_id is not None and table_id is not None:
                info_string = DctServer._create_info_string(button_id, table_id, 0, 0)
            else:
                info_string = DctServer._create_info_string(-1, -1, 0, 0)
            return DctServer.get_pareto_front(request, info_string, "/html_homepage1")
        elif action == "details" and table_id == 1:
            # Check if button_id is valid
            if button_id is not None:
                DctServer._c_config_id = button_id
            else:
                DctServer._c_config_id = 0
            return await DctServer.main_page2(request, "", 0, user)
        elif action == "control_sheet":
            # Check if user is authorized
            if user is not None:
                if DctServer.break_status == 1:
                    DctServer.break_status = 0
                else:
                    DctServer.break_status = 1
                # Display the control page
                break_status = DctServer.break_status
                return DctServer.templates.TemplateResponse("control_page.html",
                                                            {"request": request, "url_back": "/html_homepage1",
                                                             "break_status": break_status})

        # Init request for main process
        request_data = SrvReqData()
        request_data.req_cmd = ReqCmd.sd_main

        # Request data from main process
        DctServer.srv_request_queue.put(request_data)
        # Wait for response
        data: srv_ctl_dtos.QueueMainData = DctServer.srv_response_queue.get()

        # Add content circuit config
        # Create list (in future it is a list of configurations)
        circuit_conf_list: list[srv_ctl_dtos.ConfigurationDataEntryDto] = data.circuit_list
        table_data_circuit = DctServer.get_table_data(circuit_conf_list)

        # Add content heat sink config
        table_data_heat_sink = DctServer.get_heat_sink_table_data(data.heat_sink_list)
        # table_data_heat_sink = DctServer.get_table_data(data.heat_sink_list)

        # Add content summary
        table_data_summary = DctServer.get_summary_table_data(data.summary_list)

        return DctServer.templates.TemplateResponse("main_page1.html",
                                                    {"request": request, "c_table_data": table_data_circuit,
                                                     "h_table_data": table_data_heat_sink,
                                                     "s_table_data": table_data_summary,
                                                     "total_proc_time": DctServer.get_format_time(data.total_proc_time),
                                                     "total_inductor_proc_time": DctServer.get_format_time(data.inductor_proc_time),
                                                     "total_transformer_proc_time": DctServer.get_format_time(data.transformer_proc_time),
                                                     "text_message": DctServer.status_message,
                                                     "user": user})

    @staticmethod
    @app.get("/html_homepage2", response_class=HTMLResponse, response_model=None)
    async def main_page2(request: Request, action: str = "", c_filt_pt_ID: int = 0,
                         user: str | None = Depends(get_current_user),
                         button_id: int | None = None, table_id: int | None = None) -> _TemplateResponse | HTMLResponse:
        """Provide the html-information based on client request to html_homepage2.

        :param request: Request information of the client request
        :type  request: Request
        :param action: Information about the requested action (Keyword driven)
        :type  action: str
        :param c_filt_pt_ID: Index of the selected circuit filtered point
        :type  c_filt_pt_ID: int
        :param user: User, in case of valid user
        :type  user: str
        :param button_id: Index of selected button, if a button was pressed
        :type  button_id: int
        :param table_id: Index of selected table in case of a button press
        :type  table_id: int
        :return: html-page
        :rtype:  _TemplateResponse | HTMLResponse
        """
        # Check if an action is requested
        # User request: Logout
        if action == "logout":
            request.session.clear()
            user = None
            DctServer.status_message = "User is logged out"
        # User request: Button press to display Pareto-front
        elif action == "pareto_circuit":
            if button_id is not None and table_id is not None:
                info_string = DctServer._create_info_string(button_id, table_id, DctServer._c_filt_pt_ID, 1)
            else:
                info_string = DctServer._create_info_string(-1, -1, 0, 0)

            # Display the Pareto-front
            return DctServer.get_pareto_front(request, info_string, "/html_homepage2")
        # User request: Button press to change to control sheet
        elif action == "control_sheet":
            # Check if user is authorized
            if user is not None:
                if DctServer.break_status == 1:
                    DctServer.break_status = 0
                else:
                    DctServer.break_status = 1
                # Display the control page
                break_status = DctServer.break_status
                return DctServer.templates.TemplateResponse("control_page.html",
                                                            {"request": request, "url_back": "/html_homepage2",
                                                             "break_status": break_status})

        # Init request for main process
        request_data = SrvReqData()
        request_data.req_cmd = ReqCmd.sd_detail
        request_data.c_filt_pt_id = c_filt_pt_ID

        # Save the selected filtered point id
        DctServer._c_filt_pt_ID = c_filt_pt_ID

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

        # Add content heat_sink config
        table_data_heat_sink = DctServer.get_heat_sink_table_data(data.heat_sink_list)
        # table_data_heat_sink = DctServer.get_table_data(data.heat_sink_list)

        # Add content summary
        table_data_summary = DctServer.get_summary_table_data([data.summary_data])

        return DctServer.templates.TemplateResponse("main_page2.html", {"request": request,
                                                                        "c_table_data": table_data_circuit,
                                                                        "i_table_data": table_data_inductor,
                                                                        "t_table_data": table_data_transformer,
                                                                        "h_table_data": table_data_heat_sink,
                                                                        "stable_data": table_data_summary,
                                                                        "conf_proc_time": DctServer.get_format_time(data.conf_proc_time),
                                                                        "c_filt_pt_ID": c_filt_pt_ID,
                                                                        "text_message": DctServer.status_message,
                                                                        "user": user})

    @staticmethod
    @app.get("/control_page", response_class=HTMLResponse, response_model=None)
    async def control_page(request: Request, action: str = "", url_back: str = "/html_homepage1") -> _TemplateResponse | HTMLResponse:
        """Provide the html-information based on client request to control_page.

        :param request: Request information of the client request
        :type  request: Request
        :param action: Information about the requested action (Keyword driven)
        :type  action: str
        :param url_back: Uniform resource locator for jump back
        :type  url_back: str
        :return: html-page
        :rtype:  _TemplateResponse
        """
        if action == "continue":
            DctServer.status_message = "Continue is active"
        elif action == "pause":
            DctServer.status_message = "Pause is active"
        elif action == "stop":
            DctServer.status_message = "Stops the server and the optimization (if prog_exit_flag==true)"
            DctServer.req_stop.value = 1

        return DctServer.templates.TemplateResponse("control_page.html",
                                                    {"request": request, "url_back": url_back,
                                                     "break_status": DctServer.break_status})

    @staticmethod
    @app.get("/login", response_class=HTMLResponse, response_model=None)
    async def login_page(request: Request) -> _TemplateResponse:
        """Provide the html-information based on client request to login_page.

        :param request: Request information of the client request
        :type  request: Request
        :return: html-page
        :rtype:  _TemplateResponse
        """
        return DctServer.templates.TemplateResponse("login.html", {"request": request})

    @staticmethod
    @app.post("/login", response_model=None)
    async def login(request: Request, username: str = Form(...), password: str = Form(...)) -> _TemplateResponse | RedirectResponse:
        """Evaluate the login data and create a session.

        If the form send by '@app.get("/login"' send back this method is called.
        It verify the username and password and provide the suitable evaluation result.
        :param request: Request information of the client request
        :type  request: Request
        :param username: Name of the user
        :type  username: str
        :param password: password
        :type  password: str
        :return: html-page
        :rtype:  _TemplateResponse | RedirectResponse
        """
        if DctServer.users.get(username) == password:
            request.session["user"] = username
            # Send back success and request client to request base url '/' with GET-Methode (303)
            return RedirectResponse(url="/", status_code=303)
        return DctServer.templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    @staticmethod
    @app.get("/admin", response_class=HTMLResponse, response_model=None)
    async def admin_page(request: Request, username: str = Depends(get_current_user)) -> _TemplateResponse | RedirectResponse:
        """Evaluate the permission of user and return the suitable page.

        If the form send by '@app.get("/login"' send back this method is called.
        It verify the username and password and provide the suitable evaluation result.
        :param request: Request information of the client request
        :type  request: Request
        :param username: Name of the user
        :type username: str
        :return: html-page
        :rtype:  _TemplateResponse | RedirectResponse
        """
        if not username:
            return RedirectResponse(url="/login")
        return DctServer.templates.TemplateResponse("admin.html", {"request": request, "user": username})

    @staticmethod
    @app.get("/pareto_front")
    def get_pareto_front(request: Request, info_string: str = "", url_back: str = "") -> HTMLResponse:
        """Provide the Pareto-front.

        Later to replace by the selected Pareto-front
        :param request: Request information of the client request
        :type  request: Request
        :param info_string: Information to add to html-sheet
        :type  info_string: str
        :param url_back: Uniform resource locator for jump back
        :type  url_back: str
        :return: html-page
        :rtype:  HTMLResponse
        """
        # Later to replace by request of optuna Pareto front data
        original_html = DctServer.load_optuna_html_file(DctServer._optuna_path)

        # Add information and back-button
        insert_html = "<h1>Actual Pareto front</h1>" + "<p>" + info_string + "</p>"
        insert_html = insert_html + "<button onclick=\"location.href='" + url_back
        insert_html = insert_html + "'\">Back to main menue</button> "

        # Search for tag body
        body_index = original_html.lower().find("<body>")
        if body_index == -1:
            return HTMLResponse(content="Error: <body> Tag not found", status_code=500)

        # Seek position to add
        insertion_point = body_index + len("<body>")
        # Add values
        modified_html = (original_html[:insertion_point] + "\n" + insert_html + "\n" + original_html[insertion_point:])

        return HTMLResponse(content=modified_html)
