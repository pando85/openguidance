import threading
import utm
import logging

import guidancelib


class GPS(threading.Thread):
    """Thread that reads value from GPS"""
    def __init__(self, gps_queue):
    
        super(GPS, self).__init__()

        self.gps_queue = gps_queue
        self.stop_request = threading.Event()

        self.x = None
        self.y = None

    def run(self):
        logging.info('Starting GPS thread')
        if guidancelib.config['gps.input'] == 'simulate':
            self._simulation()

    def _simulation(self):
        with open(guidancelib.config['gps.simulate_file']) as simulate_file:
            gps_coordinates = self._parse_file(simulate_file.readlines())

        timestep = 0
        while not self.stop_request.isSet():
            try:
                if timestep == len(gps_coordinates):
                    timestep = 0

                self.gps_queue.put(self._get_utm(*gps_coordinates[timestep]))
                logging.debug('Position: x -> %s, y -> %s', self.x, self.y)
                timestep += 1
            except:
                raise

    def _parse_file(self, simulation):
        gps_coordinates_str = [coordinate.split(',') for coordinate in simulation]
        gps_coordinates = [[float(i) for i in coordinate ]for coordinate in gps_coordinates_str]
        return gps_coordinates

    def _get_utm(self, altitude, longitude):
        utm_coordinates = utm.from_latlon(altitude, longitude)
        self.x = utm_coordinates[0]
        self.y = utm_coordinates[1]
        return [self.x, self.y]

    def join(self, timeout=None):
        self.stop_request.set()
        super(GPS, self).join(timeout)