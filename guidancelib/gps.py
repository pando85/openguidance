import threading
import utm 

import guidancelib


class GPS(threading.Thread):
    """Thread that reads value from GPS"""
    def __init__(self, gps_queue):
    
        super(GPS, self).__init__()

        self.gps_queue = gps_queue
        self.stop_request = threading.Event()

        self.altitude = None
        self.longitude = None

    def run(self):
        if guidancelib.config['gps.input'] == 'simulate':
            self._simulation()

    def _simulation(self):
        #read from guidancelib.config['gps.simulate_file']
        # transform to utm
        # put in queue        
        with open(guidancelib.config['gps.simulate_file']) as simulate_file:
            gps_coordinates = self._parse_file(simulate_file.readlines())

        point = 0
        while not self.stop_request.isSet():
            try:
                if point == len(gps_coordinates):
                    point = 0

                self.gps_queue.put(gps_coordinates[point])
                point += 1
            except:
                raise

    def _parse_simulation(self, simulation):
        gps_coordinates_str = [coordinate.split(',') for coordinate in simulation]
        gps_coordinates = [[float(i) for i in coordinate ]for coordinate in gps_coordinates]
        return gps_coordinates

    def _get_utm(self):
        x,y, = utm.from_latlon(self.altitude, self.longitude)
        return [x,y]

    def join(self, timeout=None):
        self.stop_request.set()
        super(GPS, self).join(timeout)