#!/usr/bin/env python

""" Collects data about bookmarked stories for a list of users """

import argparse
import os
import requests
import bs4
import time

########################################################################
# Basic modules
########################################################################

def create_folder(folderName):
    """Creates a folder if it doesn't exist"""
    if not os.path.exists(folderName):
        cmd = "mkdir -p "+folderName
        os.system(cmd)

########################################################################
# Website loading functions
########################################################################

def get_webpage(link):
    """ Obtains the link from the webpage and returns it parsed using 
    BeautifulSoup. After downloading the page it rests 5 seconds due
    to website request. 
    Input -> HTML address
    Output -> Beautiful Soap parsed HTML
    """
    response = requests.get(link)
    print("downloaded:",link)
    if response.status_code != 200:
        print("WARNING: This link did not work out:",link, response.status_code)
    soup = bs4.BeautifulSoup(response.text,"lxml")
    time.sleep(5)
    return soup


parser = argparse.ArgumentParser(description="Obtain user bookmarks for AO3")
parser.add_argument("-l",dest="listUsers",action="store",required=True,help="List of users for which bookmarks want to be acquired")
parser.add_argument("-o",dest="outFileName",action="store",default="bookmarks_users.txt",help="File name where bookmarks will be stored")
parser.add_argument("-f",dest="outListFailed",action="store",default="failed_users.txt",help="File name where failed users will be stored")
parser.add_argument("-c",dest="lastUser",action="store",default=None,help="Restart point, give last user processed")
parser.add_argument("--base_html",dest="baseHTML",action="store",default="https://archiveofourown.org/",help="Main HTML page for the archive")
args = parser.parse_args()

users = [x.strip() for x in open(args.listUsers)]

if args.lastUser:
    if args.lastUser in users:
        start = users.index(args.lastUser)
    else:
        exit("The provided user name is not in the list of users")
else:
    start = 0

with open(args.outListFailed,"w") as failedUsers:
    with open(args.outFileName,"w") as outfile:
        for userNum in range(start,len(users)):
            userName = users[userNum]
            try:
                link = args.baseHTML+"/users/"+userName+"/bookmarks"
                soup = get_webpage(link)
                try:
                    number_of_pages = int(str(soup.find_all(class_="pagination actions")[1].find_all("a")[-2]).split(">")[1].split("<")[0])
                except:
                    number_of_pages = 1
                fanficIDs = set([])
                if number_of_pages == 1:
                    bookmarks = soup.find_all(class_="bookmark blurb group")
                    for book in bookmarks: 
                        counts = soup.find_all(class_="count") 
                        for c in counts: 
                            t = c.find("a").get("href").split("/")[2] 
                            fanficIDs.add(t)
                else:
                    for numPage in range(1,number_of_pages + 1):
                        linkPage = link+"?page="+str(numPage)
                        soup = get_webpage(linkPage)
                        bookmarks = soup.find_all(class_="bookmark blurb group")
                        for book in bookmarks: 
                            counts = soup.find_all(class_="count") 
                            for c in counts: 
                                t = c.find("a").get("href").split("/")[2] 
                                fanficIDs.add(t)
                if len(fanficIDs) != 0:
                    print(userName+"\t"+"\t".join(list(fanficIDs)),file=outfile)
            except:
                print(userName,outfile=failedUsers)

