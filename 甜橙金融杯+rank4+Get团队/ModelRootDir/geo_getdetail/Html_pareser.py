from bs4 import BeautifulSoup
import re

class HtmlPare(object):
    def getdata(self, data):
        return data
        # print("find fans elements.....")
        # getnodes = re.findall(r"<script>+.*</script>+",str(souprep),re.I|re.M)
        # for item in getnodes:
        #     if re.match(r'.*粉丝列表.*', str(item)):
        #         return item
