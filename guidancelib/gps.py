import threading
import utm
import logging
import pynmea2
import serial

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
        elif guidancelib.config['gps.input'] == 'serial':
            self._serial()


    def _serial(self):
        logging.info('Starting GPS serial connection')
        try:
            logging.info('Try open: %s', guidancelib.config['gps.port'])
            gps_serial = serial.Serial(guidancelib.config['gps.port'])
            gps_serial.baudrate = 115200 
            gps_serial.bytesize = 8      
            gps_serial.parity   = 'N'    
            gps_serial.stopbits = 1

            while not self.stop_request.isSet():
                gps_nmea = pynmea2.parse(gps_serial.readline())
                self.gps_queue.put(self._get_utm([gps_nmea.latitude, gps_nmea.longitude]))
                logging.debug('Position: x -> %s, y -> %s', self.x, self.y)
            
            logging.debug('Closing serial connection')
            gps_serial.close()

        except:
            sleep(1)
            self._serial()

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

    def _parse_file(self, simulation_file):
        gps_nmea = [pynmea2.parse(line) for line in simulation_file]
        gps_coordinates = [[gps_nmea_line.latitude, gps_nmea_line.longitude] for gps_nmea_line in gps_nmea]
        return gps_coordinates

    def _get_utm(self, altitude, longitude):
        utm_coordinates = utm.from_latlon(altitude, longitude)
        self.x = utm_coordinates[0]
        self.y = utm_coordinates[1]
        return [self.x, self.y]

    def join(self, timeout=None):
        self.stop_request.set()
        super(GPS, self).join(timeout)