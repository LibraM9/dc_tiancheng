import geohash
class UrlManager(object):
    def __init__(self):
        self.new_urls = set()
        self.old_urls = set()
    def getTotalLen(self):
        return len(self.new_urls)
    def getFinished(self):
        return len(self.old_urls)
    def makecordi(self, geostrSets):
        for geoStr in geostrSets:
            # print(geoStr)
            if str(geoStr) == "nan":
                continue
            latitudeAndlongitude = geohash.decode(geoStr)
            # print(latitudeAndlongitude)

            latitude,longitude = latitudeAndlongitude[0],latitudeAndlongitude[1]
            url = "https://apis.map.qq.com/ws/geocoder/v1/?location=%s,%s&get_poi=1&key=M4DBZ-OX2W4-OUEU6-XD64O-WNSBZ-7JFTL"%(latitude,longitude)
            paraCoUrl = geoStr + "|"+ latitude + "|"+longitude + "|" + url
            self.new_urls.add(paraCoUrl)              

    def hasNewUrl(self):
        return len(self.new_urls) != 0
 
    def getNewUrl(self):
         new_url = self.new_urls.pop()
         self.old_urls.add(new_url)
         return new_url


