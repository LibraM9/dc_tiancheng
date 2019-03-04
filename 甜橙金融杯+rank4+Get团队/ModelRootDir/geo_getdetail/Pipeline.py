import re

class pinline(object):
	def __init__(self):
		pass
	def getdata(self,data):
		dataTodic = {}
		dataTodic["geo_code"] = data[0]
		dataTodic["latitude"] = data[1]
		dataTodic["longitude"] = data[2]

		dataTodic["nation"] = data[3]["nation"]
		dataTodic["province"] = data[3]["province"]
		dataTodic["city"] = data[3]["city"]
		dataTodic["district"] = data[3]["district"]
		dataTodic["street"] = data[3]["street"]
		dataTodic["street_number"] = data[3]["street_number"]
		return dataTodic
