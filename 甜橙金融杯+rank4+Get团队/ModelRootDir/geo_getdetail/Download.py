from urllib import request
import time

class Downloader(object):
    def __init__(self):
        pass
    @staticmethod
    def getJSON(urlobj):
        # print("\ndownload url : %s......\n"%urlobj)
        geoStr,lon,lat,url = urlobj.split("|")
        res = request.Request(url)
        response = request.urlopen(res)
        if response.getcode() != 200:
            print("err" + urlobj)
            return None
        responsedata =  eval((response.read().decode("utf-8")))

        if "result" not in responsedata.keys():
            return None
            
        result = responsedata["result"]
        if "address_component" not in result.keys():
            return None
        address_component =result["address_component"]
        msgKeys = ["nation","province","city","district","street","street_number"]
        if "nation" not in address_component.keys():
            return None
        if address_component["nation"] != "中国":
            tmp_address_component = {}
            tmp_address_component["nation"] = address_component["nation"]
            for k in msgKeys[1:]:
                tmp_address_component[k] = "unknow"
            address_component = tmp_address_component
        for k in msgKeys:
            if k not in address_component.keys():
                print(address_component.keys())
                return None
        return [geoStr,lon,lat,address_component]

    def geturl(self, urlobj):
        # print("in Downloader...............")
        maxTime = 3
        retryTime = 0
        while retryTime <= maxTime:
            try:
                data = Downloader.getJSON(urlobj)
            except Exception as e:
                data = None
            
            if data is None:
                print("retring getting data in ===> \n################\n|\n|  %s \n|\n|\n################" % urlobj)
                retryTime = retryTime + 1
            else:
                break

        # print("#################################\n\\n\n\n\n\n\n")
        
        time.sleep(0.5)
        # print(result["address_component"])
