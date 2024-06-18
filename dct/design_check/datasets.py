"""Describe the dataclasses how to store the input parameters and results."""
import os
import pprint
import numpy as np
from dotmap import DotMap

from dct.debug_tools import *
import transistordatabase as tdb


class DabData(DotMap):
    """
    Class to store the DAB specification, modulation and simulation results and some more data.

    It contains only numpy arrays, you can only add those.
    In fact everything is a numpy array even single int or float values!
    It inherits from DotMap to provide dot-notation usage instead of regular dict access.
    Make sure your key names start with one of the "_allowed_keys", if not you can not add the key.
    Add a useful name string after the prefix from "_allowed_keys" to identify your results later.
    """

    _allowed_keys = ['_timestamp', '_comment', 'spec_', 'mesh_', 'mod_', 'sim_', 'meas_', 'coss_', 'qoss_', 'iter_']
    _allowed_spec_keys = ['V1_nom', 'V1_min', 'V1_max', 'V1_step', 'V2_nom', 'V2_min', 'V2_max', 'V2_step', 'P_min',
                          'P_max', 'P_nom', 'P_step', 'n', 'Ls', 'Lm', 'Lc1', 'Lc2', 'Lc2_', 'fs', 't_dead1', 't_dead2',
                          'temp', 'C_HB11', 'C_HB12', 'C_HB21', 'C_HB22']

    def __init__(self, *args, **kwargs):
        """
        Init the DabData class.

        Initialisation with another Dict is not handled and type converted yet.

        :param args:
        :param kwargs:
        """
        if args or kwargs:
            warning("Don't use this type of initialisation!")
        # if kwargs:
        #     d.update((k, float(v)) for k,v in self.__call_items(kwargs)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        """
        Set elements to this class.

        Only np.ndarray is allowed
        """
        if isinstance(value, np.ndarray):
            # Check for allowed key names
            if any(key.startswith(allowed_key) for allowed_key in (self._allowed_keys + self._allowed_spec_keys)):
                super().__setitem__(key, value)
            else:
                warning('None of the _allowed_keys are used! Nothing added! Used key: ' + str(key))
        else:
            # Value will be converted to a ndarray
            # Check for allowed key names
            if any(key.startswith(allowed_key) for allowed_key in (self._allowed_keys + self._allowed_spec_keys)):
                super().__setitem__(key, np.asarray(value))
            else:
                warning('None of the _allowed_keys are used! Nothing added! Used key: ' + str(key))

    def pprint_to_file(self, filename):
        """
        Print the DAB in nice human-readable form into a text file.

        WARNING: This file can not be loaded again! It is only for documentation.
        :param filename:
        """
        filename = os.path.expanduser(filename)
        filename = os.path.expandvars(filename)
        filename = os.path.abspath(filename)
        if os.path.isfile(filename):
            warning("File already exists!")
        else:
            with open(filename, 'w') as file:
                pprint.pprint(self.toDict(), file)

    def save_to_file(self, directory=str(), name=str(), timestamp=True, comment=str()):
        """
        Save everything (except plots) in one file.

        WARNING: Existing files will be overwritten!

        File is ZIP compressed and contains several named np.ndarray objects:
            # String is constructed as follows:
            # used module (e.g. "mod_sps_") + value name (e.g. "phi")
            mod_sps_phi: mod_sps calculated values for phi
            mod_sps_tau1: mod_sps calculated values for tau1
            mod_sps_tau2: mod_sps calculated values for tau1
            sim_sps_iLs: simulation results with mod_sps for iLs
            sim_sps_S11_p_sw:

        :param directory: Folder where to save the files
        :param name: String added to the filename. Without file extension. Datetime may prepend the final name.
        :param timestamp: If the datetime should prepend the final name. default True
        :param comment:
        """
        # Add some descriptive data to the file
        # Adding a timestamp, it may be useful
        self['_timestamp'] = np.asarray(datetime.now().isoformat())
        # Adding a comment to the file, hopefully a descriptive one
        if comment:
            self['_comment'] = np.asarray(comment)

        # Adding a timestamp to the filename if requested
        if timestamp:
            if name:
                filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
            else:
                filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        else:
            if name:
                filename = name
            else:
                # set some default non-empty filename
                filename = "dab_dataset"

        if directory:
            directory = os.path.expanduser(directory)
            directory = os.path.expandvars(directory)
            directory = os.path.abspath(directory)
            if os.path.isdir(directory):
                file = os.path.join(directory, filename)
            else:
                warning("Directory does not exist!")
                file = os.path.join(filename)
        else:
            file = os.path.join(filename)

        # numpy saves everything for us in a handy zip file
        np.savez_compressed(file=file, **self)

    def gen_meshes(self):
        """
        Generate the default meshgrids for V1, V2 and P.

        Values for:
        'V1_nom', 'V1_min', 'V1_max', 'V1_step',
        'V2_nom', 'V2_min', 'V2_max', 'V2_step',
        'P_min', 'P_max', 'P_nom', 'P_step'
        must be set first!
        """
        self.mesh_V1, self.mesh_V2, self.mesh_P = np.meshgrid(np.linspace(self.V1_min, self.V1_max, int(self.V1_step)),
                                                              np.linspace(self.V2_min, self.V2_max, int(self.V2_step)),
                                                              np.linspace(self.P_min, self.P_max, int(self.P_step)),
                                                              sparse=False)

    def append_result_dict(self, result: dict, name_pre: str = '', name_post: str = ''):
        """Unpack the results and append it to the result dictionary."""
        for key, value in result.items():
            self[name_pre + key + name_post] = value

    def import_c_oss_from_file(self, file: str, name: str):
        """
        Import a csv file containing the Coss(Vds) capacitance from the MOSFET datasheet.

        This may be generated with: https://apps.automeris.io/wpd/

        Note we assume V_ds in Volt and C_oss in F. If this is not the case, scale your data accordingly!

        CSV File should look like this:
        # V_ds / V; C_oss / F
        1,00; 900,00e-12
        2,00; 800,00e-12
        :param file: csv file path
        :type file: str
        :param name: transistor name
        :type name: str
        """
        file = os.path.expanduser(file)
        file = os.path.expandvars(file)
        file = os.path.abspath(file)

        # Conversion from decimal separator comma to point so that np can read floats
        # Be careful if your csv is actually comma separated! ;)
        def conv(x):
            return x.replace(',', '.').encode()

        # Read csv file
        csv_data = np.genfromtxt((conv(x) for x in open(file)), delimiter=';', dtype=float)

        # Maybe check if data is monotonically
        # Check if voltage is monotonically rising
        if not np.all(csv_data[1:, 0] >= csv_data[:-1, 0], axis=0):
            warning("The voltage in csv file is not monotonically rising!")
        # Check if Coss is monotonically falling
        if not np.all(csv_data[1:, 1] <= csv_data[:-1, 1], axis=0):
            warning("The C_oss in csv file is not monotonically falling!")

        # Rescale and interpolate the csv data to have a nice 1V step size from 0V to v_max
        # A first value with zero volt will be added
        v_max = int(np.round(csv_data[-1, 0]))
        v_interp = np.arange(v_max + 1)
        coss_interp = np.interp(v_interp, csv_data[:, 0], csv_data[:, 1])
        # Since we now have a evenly spaced vector where x corespond to the element-number of the vector
        # we don't have to store x (v_interp) with it.
        # To get Coss(V) just get the array element coss_interp[V]

        # np.savetxt('coss_' + name + '.csv', coss_interp, delimiter=';')

        # return coss_interp
        self['coss_' + name] = coss_interp
        self['qoss_' + name] = self._integrate_c_oss(coss_interp)

    def import_c_oss_from_tdb(self, transistor: tdb.Transistor):
        """
        Import the transistor Coss(Vds) capacitance from the transistor database (TDB).

        Note we assume V_ds in Volt and C_oss in F. If this is not the case, scale your data accordingly!

        :param file: csv file path
        :type file: str
        :param name: transistor name
        :type name: str
        """
        csv_data = transistor.c_oss[0].graph_v_c.T

        # Maybe check if data is monotonically
        # Check if voltage is monotonically rising
        if not np.all(csv_data[1:, 0] >= csv_data[:-1, 0], axis=0):
            warning("The voltage in csv file is not monotonically rising!")
        # Check if Coss is monotonically falling
        if not np.all(csv_data[1:, 1] <= csv_data[:-1, 1], axis=0):
            warning("The C_oss in csv file is not monotonically falling!")

        # Rescale and interpolate the csv data to have a nice 1V step size from 0V to v_max
        # A first value with zero volt will be added
        v_max = int(np.round(csv_data[-1, 0]))
        v_interp = np.arange(v_max + 1)
        coss_interp = np.interp(v_interp, csv_data[:, 0], csv_data[:, 1])
        # Since we now have a evenly spaced vector where x corespond to the element-number of the vector
        # we don't have to store x (v_interp) with it.
        # To get Coss(V) just get the array element coss_interp[V]

        # np.savetxt('coss_' + name + '.csv', coss_interp, delimiter=';')

        # return coss_interp
        self['coss_' + transistor.name] = coss_interp
        self['qoss_' + transistor.name] = self._integrate_c_oss(coss_interp)

    def _integrate_c_oss(self, coss):
        """
        Integrate Coss for each voltage from 0 to V_max.

        :param coss: MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
        :return: Qoss(Vds) as one row of data and index = Vds.
        """

        # Integrate from 0 to v
        def integrate(v):
            v_interp = np.arange(v + 1)
            coss_v = np.interp(v_interp, np.arange(coss.shape[0]), coss)
            return np.trapz(coss_v)

        coss_int = np.vectorize(integrate)
        # get an qoss vector that has the resolution 1V from 0 to V_max
        v_vec = np.arange(coss.shape[0])
        # get an qoss vector that fits the mesh_V scale
        # v_vec = np.linspace(V_min, V_max, int(V_step))
        qoss = coss_int(v_vec)
        # Scale from pC to nC
        # qoss = qoss / 1000

        # np.savetxt('qoss.csv', qoss, delimiter=';')
        return qoss


def save_to_file(dab: DabData, directory=str(), name=str(), timestamp=True, comment=str()):
    """
    Save everything (except plots) in one file.

    WARNING: Existing files will be overwritten!

    File is ZIP compressed and contains several named np.ndarray objects:
        # String is constructed as follows:
        # used module (e.g. "mod_sps_") + value name (e.g. "phi")
        mod_sps_phi: mod_sps calculated values for phi
        mod_sps_tau1: mod_sps calculated values for tau1
        mod_sps_tau2: mod_sps calculated values for tau1
        sim_sps_iLs: simulation results with mod_sps for iLs
        sim_sps_S11_p_sw:

    :param comment:
    :param dab:
    :param directory: Folder where to save the files
    :param name: String added to the filename. Without file extension. Datetime may prepend the final name.
    :param timestamp: If the datetime should prepend the final name. default True
    """
    # Add some descriptive data to the file
    # Adding a timestamp, it may be useful
    dab['_timestamp'] = np.asarray(datetime.now().isoformat())
    # Adding a comment to the file, hopefully a descriptive one
    if comment:
        dab['_comment'] = np.asarray(comment)

    # Adding a timestamp to the filename if requested
    if timestamp:
        if name:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
        else:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        if name:
            filename = name
        else:
            # set some default non-empty filename
            filename = "dab_dataset"

    if directory:
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            file = os.path.join(directory, filename)
        else:
            warning("Directory does not exist!")
            file = os.path.join(filename)
    else:
        file = os.path.join(filename)

    # numpy saves everything for us in a handy zip file
    np.savez_compressed(file=file, **dab)


def load_from_file(file: str) -> DabData:
    """
    Load everything from the given .npz file.

    :param file: a .nps filename or file-like object, string, or pathlib.Path
    :return: two objects with type DAB_Specification and DAB_Results
    """
    dab = DabData()
    # Check for filename extension
    file_name, file_extension = os.path.splitext(file)
    if not file_extension:
        file += '.npz'
    file = os.path.expanduser(file)
    file = os.path.expandvars(file)
    file = os.path.abspath(file)
    # Open the file and parse the data
    with np.load(file) as data:
        for k, v in data.items():
            dab[k] = v
    return dab


def save_to_csv(dab: DabData, key=str(), directory=str(), name=str(), timestamp=True):
    """
    Save one array with name 'key' out of dab_results to a csv file.

    :param dab:
    :param key: name of the array in dab_results
    :param directory:
    :param name: filename without extension
    :param timestamp: if the filename should prepended with a timestamp
    """
    if timestamp:
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
    filename = key + '_' + name + '.csv'

    if directory:
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            file = os.path.join(directory, filename)
        else:
            warning("Directory does not exist!")
            file = os.path.join(filename)
    else:
        file = os.path.join(filename)

    comment = key + ' with P: {}, V1: {} and V2: {} steps.'.format(
        int(dab.P_step),
        int(dab.V1_step),
        int(dab.V2_step)
    )

    # Write the array to disk
    with open(file, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# ' + comment + '\n')
        outfile.write('# Array shape: {0}\n'.format(dab[key].shape))
        # x: P, y: V1, z(slices): V2
        outfile.write('# x: P ({}-{}), y: V1 ({}-{}), z(slices): V2 ({}-{})\n'.format(
            int(dab.P_min),
            int(dab.P_max),
            int(dab.V1_min),
            int(dab.V1_max),
            int(dab.V2_min),
            int(dab.V2_max)
        ))
        outfile.write('# z: V2 ' + np.array_str(dab.mesh_V2[:, 0, 0], max_line_width=10000) + '\n')
        outfile.write('# y: V1 ' + np.array_str(dab.mesh_V1[0, :, 0], max_line_width=10000) + '\n')
        outfile.write('# x: P ' + np.array_str(dab.mesh_P[0, 0, :], max_line_width=10000) + '\n')

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        i = 0
        for array_slice in dab[key]:
            # Writing out a break to indicate different slices...
            outfile.write('# V2 slice {}V\n'.format(
                (dab.V2_min + i * (dab.V2_max - dab.V2_min) / (dab.V2_step - 1))
            ))
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            # np.savetxt(outfile, array_slice, fmt='%-7.2f')
            np.savetxt(outfile, array_slice, delimiter=';')
            i += 1

def show_keys(dab: DabData) -> None:
    """
    Show the keys, stored in the given DabData structure.

    :param dab: DabData object
    :return: None
    """
    for key, value in dab.items():
        print(f"{key=}")