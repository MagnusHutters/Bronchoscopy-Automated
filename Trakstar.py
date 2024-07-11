import os
from trakstar import TrakSTARInterface

#TODO missing ip connection details

trakstar = None
exp = None
udp_connection = None

def initialize():
    """returns remote_control and file

     If remote_control or filename is None, the function will ask it (user input)

     """

    global trakstar, udp_connection, exp
    trakstar = TrakSTARInterface()
    trakstar.initialize()

    if trakstar.is_init:
        udp_connection = trakstar.udp
        print("Trakstar initialized")
    else:
        print("No init")

def end():
    if trakstar is not None:
        #logo_text_line(text="Closing trakSTAR").present()
        trakstar.close_data_file()
        trakstar.close()

def prepare_recoding():
    """get filename, allow changing of settings
    and make connection in remote control mode
    """
    # todo: display configuration
    if trakstar is None:
        raise RuntimeError("Pytrak not initialized")
    
    #set system settings
    trakstar.set_system_configuration(
                            measurement_rate = 80,
                            max_range = 36,
                            power_line = 60,
                            metric = True,
                            report_rate = 1,
                            print_configuration = True)

    comment_str = "Motion tracking data recorded with " + \
                   "Pytrak "
    '''trakstar.open_data_file(filename = filename,
                            directory = "data",
                            suffix = ".csv",
                            time_stamp_filename = False,
                            write_angles = True,
                            write_quality = True,
                            write_udp = None,
                            write_cpu_times = True,
                            comment_line = comment_str)'''

def record_data():
    if trakstar is None:
        raise RuntimeError("Pytrak not initialized")
    quit_recording = False

    m_countBO = 0
    while not quit_recording:
        # get data and process
        data_array = trakstar.get_synchronous_data_dict(write_data_file = False)
        
        print(data_array)
        m_countBO = m_countBO + 1
        if(m_countBO > 5000):
            quit_recording = True

def run():
    global trakstar, exp, udp_connection
    print("Pytrak")
    initialize()
    if not trakstar.is_init:
        end()
    prepare_recoding(filename="ola")
    record_data()
    end()






class Trakstar:
    def __init__(self):
        global trakstar, exp, udp_connection
        initialize()

        if not trakstar.is_init:
            end()

        prepare_recoding()



    def trakstarFormat(self, data):

        newData = {}

        newData["time"] = data["time"]
        newData["cpu_time"] = data["cpu_time"]
        newData["camera"] = data[1].tolist()
        newData["prope"] = data[2].tolist()
        newData["box"] = data[3].tolist()

        newData["udp"] = data["udp"]


        return newData


    def getFrame(self):
        global trakstar, exp, udp_connection
        if not trakstar.is_init:
            end()
            return None
        data_array = trakstar.get_synchronous_data_dict(write_data_file = False)
        data_array = self.trakstarFormat(data_array)
        return data_array


    def close(self):
        global trakstar, exp, udp_connection
        end()

    def __del__(self):
        self.close()

if __name__ == "__main__":
    
    
    trackstar = Trakstar()
    for i in range(100):
        print(trackstar.getFrame())
    
    trackstar.close()
    #run()





