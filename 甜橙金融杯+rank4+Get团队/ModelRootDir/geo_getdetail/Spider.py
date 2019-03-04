import Download,FeeDExport,Html_pareser,Pipeline,URLSchedul
import sys
import os
import numpy as np # for linear algebra
import datetime # to dela with date and time
import pandas as pd
import progressbar


class SpiderMain(object):
    def __init__(self):
        self.UrlAndIDContr = URLSchedul.UrlManager()
        self.downloader = Download.Downloader()
        self.parser = Html_pareser.HtmlPare()
        self.ProceClean = Pipeline.pinline()
        self.outjson = FeeDExport.FeedExp()
        self.CollectAllData={}
        self.errGeoGet = []
        msgKeys = ["geo_code","latitude","longitude","nation","province","city","district","street","street_number"]
        for k in msgKeys:
            self.CollectAllData[k] = []
    def genURL(self, genStrSet):
        self.UrlAndIDContr.makecordi(genStrSet)



    def craw(self):

        ###opts  为1时判断是否还有待爬用户对象，2时判断是否还有待爬URL
        maxval = self.UrlAndIDContr.getTotalLen()

        bar = progressbar.ProgressBar(maxval= maxval, 
         widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        while self.UrlAndIDContr.hasNewUrl():
            # print("preparing craw pages.....\n\n ")
            newurl = self.UrlAndIDContr.getNewUrl()
            getFinished = self.UrlAndIDContr.getFinished()
            bar.update(getFinished)

            responses = self.downloader.geturl(newurl)
            if responses is None:
                self.errGeoGet.append(newurl)
                print("responses is None")
                geoStr,lon,lat,url = newurl.split("|")

                print("err URL \n################\n|\n|  %s \n|\n|\n################"%url)
                continue
            resData = self.parser.getdata(responses)
            poroced = self.ProceClean.getdata(resData)
   
            for k,v in poroced.items():
                self.CollectAllData[k].append(v)


            # [geoStr,lon,lat,response.read().decode("utf-8")["result"]]
        bar.finish()
            
 
            
        self.outjson.tojson(self.CollectAllData)
        if self.errGeoGet:
            errGeos = "\n".join(self.errGeoGet)
            f = open("errGeo.txt","w")
            f.write(errGeos)
            f.close()
        print("||done||")


if __name__ == '__main__':
    obj_spider = SpiderMain()
    print("reading data ............. ")
    f = open("geo.txt","r")
    geoLs = f.readlines()
    geoLs = [i.replace("\n","") for i in geoLs]
    f.close()

    print("finshed ............. ")
    print("merging data ............. ")
    print("finshed ............. ")

    genClos = set(geoLs)
    obj_spider.genURL(genClos)
    obj_spider.craw()
