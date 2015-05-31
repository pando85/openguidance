import utm 


class GPS(object):
    def __init__(self):
        self.altitude = None
        self.longitude = None

    def get_utm(self):
        x,y, = utm.from_latlon(self.altitude, self.longitude)
        return [x,y]