#!/usr/bin/env python
import urllib
import xml.sax
import os

accessToken = '---'

class GroupHandler( xml.sax.ContentHandler ):
    def __init__(self, uid, sex, city):
        self.currentData = "" 
        self.uid = uid
        self.sex = sex
        self.city = city
        self.block = 0
        self.f = ""

    def startElement(self, tag, attributes):
        self.currentData = tag

    def endElement(self, tag):
        self.currentData = ""

    def characters(self, content):
        if self.currentData == "count":
            if int(content) < 20:
                self.block = 1
            else:
                self.f = open(self.uid + ".user","w")
                self.f.write(self.uid + " " + self.sex + " " + self.city + " " + content)
        elif self.currentData == "gid" and self.block == 0:
            self.f.write(" " + content)
            # count should always come before group list (gid).
            
    def closeStream(self):
        if self.block == 0:
            self.f.close()

def loadGroups(uid, sex, city):
    global accessToken 
    try:
        groupPage = urllib.urlopen \
        ("https://api.vk.com/method/groups.get.xml?user_id=" + \
        str(uid) + "&v=5.42&access_token=" + accessToken) #IOError
        content = groupPage.read()
        f = open("groups.get."+uid+".xml", "w")
        content = content.decode("utf-8").encode('ascii','ignore').decode("utf-8")
        f.write(content)
        f.close()
        parser = xml.sax.make_parser()
        gh = GroupHandler(uid, sex, city)
        parser.setContentHandler( gh )
        parser.parse(open("groups.get."+uid+".xml","r"))
        gh.closeStream()
        os.remove("groups.get."+uid+".xml")
    except IOError:
        pass
    

class UserHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.currentData = "" 
        self.uid = "0"
        self.sex = "0" #Not defined. 1 shows female and 2 shows male.
        self.country = "0" #Not defined. 1 shows Russia and 2 shows Ukraine.
        self.city = "0" #City, 314 is Kiev.
        self.countryOpened = False
        self.cityOpened = True

    def startElement(self, tag, attributes):
        self.currentData = tag
        if tag == "user":
            self.uid = "0"
            self.sex = "0"
            self.country = "0"
            self.city = "0"
        if tag == "city":
            self.cityOpened = True
        if tag == "country":
            self.countryOpened = True

    def endElement(self, tag):
        self.currentData = ""
        if tag == "city":
            self.cityOpened = False
        if tag == "country":
            self.countryOpened = False
        if tag == "user":
            if self.uid != "0" and self.sex != "0" and self.country == "2" and self.city != "0":
                #print self.uid, self.sex, self.country, self.city
                loadGroups(self.uid, self.sex, self.city)

    def characters(self, content):
        if self.currentData == "sex":
            self.sex = content
        elif self.currentData == "id":
            if self.cityOpened == True:
                self.city = content
            elif self.countryOpened == True:
                self.country = content
            else:
                self.uid = content
                


for i in xrange(start, end, 200): #Insert numbers for 'start' and 'end'
    if i % 10000 == 0:
        print str(i) + " reached."
    try:
        userPage = urllib.urlopen \
        ("https://api.vk.com/method/users.get.xml?user_ids=id" + \
        ",".join(map(str,range(i,i+200))) + \
        "&fields=sex,country,city&v=5.42")
        content = userPage.read()
        f = open("users.get."+str(i)+".xml", "w")
        content = content.decode("utf-8").encode('ascii','ignore').decode("utf-8")
        f.write(content)
        f.close()
        parser = xml.sax.make_parser()
        parser.setContentHandler( UserHandler() )
        parser.parse(open("users.get."+str(i)+".xml","r"))
    except IOError:
        pass