import datetime
import json
import pandas as pd
#"ABC"+"_"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class FeedExp(object):
	"""docstring for FeedExp"""
	# def __init__(self):

	def tojson(self, json_data):
		storeDf = pd.DataFrame()
		for k,v in json_data.items():
			storeDf[k] = v
		storeDf.to_csv("GeoDetails.csv",encoding = "utf8", index = False)
# download url : wx1b|39|115|......