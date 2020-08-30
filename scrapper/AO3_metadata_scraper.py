#!/usr/bin/env python

import requests
import bs4
import time
import argparse
import os

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

def load_webpage_from_file(fileName):
    """ Loads a pre-downloaded webpage into memory using Beautiful soup """
    with open(fileName) as infile:
        soup = bs4.BeautifulSoup(infile,"lxml")
    return soup

########################################################################
# Website parsing functions
########################################################################

def get_kudos(soup):
    """ Gets the list of users that have given a kudo for a particular
    fanfiction 
    Input -> soap from the html
    Output -> set giving the list of registered authors that gave a kudo
    """
    all_links = [f.find_all("a") for f in soup.find_all(class_="kudos")]
    published_date = soup.find_all(class_="published")[1].get_text()
    kudos = set([])
    for dades in all_links: 
        if len(dades) != 0: 
            for f in dades: 
                userName = f.get_text() 
                if userName !=  "(collapse)" and " " not in userName: 
                    kudos.add(userName)
    return kudos,published_date

def parse_comments(soup,comments,authorName):
    """ Obtains a dictionary which relates users and comments for a 
    given story. Comments are divided in two classes"""
    for comment in soup.find_all(class_="comment group even"): 
        message = comment.find(class_="userstuff").get_text() 
        writter = comment.find("a").get_text() 
        if writter != authorName:
            if writter not in comments:
                comments[writter] = []
            comments[writter].append(message)
        
    for comment in soup.find_all(class_="comment group odd"): 
        message = comment.find(class_="userstuff").get_text() 
        writter = comment.find("a").get_text()
        if writter != authorName: 
            if writter not in comments:
                comments[writter] = []
            comments[writter].append(message)
    return comments

def get_comments(soup, idName, authorName):
    """ Parses the comments and saves them by user. Multiple comments
    by a single user are collapsed into one single string """
    number_of_comment_pages = soup.find_all(class_="pagination actions")
    if len(number_of_comment_pages) != 0:
        number_of_comment_pages = str(soup.find_all(class_="pagination actions")[1].\
            find_all("a")[-2]).split(">")[1].split("<")[0]
        comment_pages = ["https://archiveofourown.org/works/"+idName+\
            "?page="+str(a)+"&show_comments=true&view_full_work=true#comments"\
             for a in range(1,int(number_of_comment_pages)+1)]
    else:
        #The first page is already loaded, so no need to save it
        comment_pages = []
    comments = {}
    #Parse first page of comments
    comments = parse_comments(soup,comments,authorName)
    #Parse the remaining pages if present
    if len(comment_pages) > 1:
        for page in comment_pages[1:]:
            soup = get_webpage(page)
            comments = parse_comments(soup,comments,authorName)
    return comments

def get_metadata(soup,baseHTML):
    """ Obtains the metadata for each story in a page. and returns a 
    dictionary of it."""
    works = soup.find_all(class_="work blurb group")
    metadata = {}
    for tag in works: 
        #Basic information:
        idName = tag.find("a").get('href').split("/")[-1]
        metadata[idName] = {}
        #Some fics are apparently annonymous:
        try:
            metadata[idName]["author"] = tag.find_all("a",rel="author")[0].get_text()
        else:
            metadata[idName]["author"] = None
        metadata[idName]["title"] = tag.find("a").get_text()
        metadata[idName]["date_update"] = tag.find(class_="datetime").get_text()
        metadata[idName]["published_date"] = metadata[idName]["date_update"]
        series = tag.find(class_="series")
        if series:
            metadata[idName]["series"] = " ".join(series.get_text().split())
        else:
            metadata[idName]["series"] = "-"
        metadata[idName]["warnings"] = "|".join([f.get_text() for f in tag.find_all(class_="warnings")])
        numWords = [f.get_text() for f in tag.find_all(class_="words")]
        if len(numWords) > 1: numWords = numWords[-1]
        metadata[idName]["numWords"] = numWords
        numChapters = [f.get_text() for f in tag.find_all(class_="chapters")]
        if len(numChapters) > 1: numChapters = numChapters[-1].split("/")[0]
        metadata[idName]["numChapters"] = numChapters
        
        # Tags:
        metadata[idName]["fandoms"] = "|".join([f.get_text() \
            for f in tag.find_all(class_="fandoms heading")[0].find_all("a",class_="tag")])
        metadata[idName]["required_tags"] = "|".join([f.get_text() \
            for f in tag.find_all(class_="required-tags")[0].find_all("a")])
        metadata[idName]["relationships"] = "|".join([f.get_text() \
            for f in tag.find_all(class_="relationships")])
        metadata[idName]["characters"] = "|".join([f.get_text()\
            for f in tag.find_all(class_="characters")])
        metadata[idName]["additional_tags"] = "|".join([f.get_text()\
            for f in tag.find_all(class_="freeforms")])
        
        
        #Aprovals
        numHits = [f.get_text() for f in tag.find_all(class_="hits")]
        if len(numHits) > 1: numHits = numHits[-1]
        metadata[idName]["numHits"] = numHits
        numKudos = [f.get_text() for f in tag.find_all(class_="kudos")]
        if len(numKudos) > 1: 
            numKudos = numKudos[-1]
            linkKudos = tag.find_all(class_="kudos")[1].find_all("a")
            linkKudos = baseHTML+linkKudos[0].get('href')
        else:
            numKudos = "0"
            linkKudos = None
        metadata[idName]["numKudos"] = numKudos
        metadata[idName]["linkKudos"] = linkKudos
        numBookmarks = [f.get_text() for f in tag.find_all(class_="bookmarks")]
        if len(numBookmarks) > 1: 
            numBookmarks = numBookmarks[-1]
        else:
            numBookmarks = "0"
        metadata[idName]["numBookmarks"] = numBookmarks
        numComments = [f.get_text() for f in tag.find_all(class_="comments")]
        if len(numComments) > 1: 
            numComments = numComments[-1]
            linkComments = baseHTML+"works/"+idName+"?show_comments=true&view_full_work=true#comments"
        else:
            numComments = "0"
            linkComments = None
        metadata[idName]["numComments"] = numComments
        metadata[idName]["linkComments"] = linkComments
    return metadata

########################################################################
# Saving functions
########################################################################

def save_metadata(metadata,allKudos,outfileName):
    """ Prints metadata into a file. This is done for every page so that
     if the script crashes it does not need to start from scratch
     Input -> metadata dictionary, list of users that left kudos
     name of the outfile where results will be printed
     No output is expected
     """
    tags = ["author","title","published_date","date_update","series","numWords",\
    "numChapters","warnings","fandoms","required_tags","relationships",\
    "characters","additional_tags","numHits","numKudos","numBookmarks",\
    "numComments","readers_kudos"]
    with open(outfileName,"w") as outfile:
        print("idName\t"+"\t".join(tags),file=outfile)
        for idName in metadata:
            #I will ignore things written by anonymous people
            if metadata[idName]["author"]:
                string = idName
                for t in tags[:-1]:
                    string += "\t"+metadata[idName][t]
                if idName in allKudos:
                    string += "\t"+"|".join(list(allKudos[idName]))
                else:
                    string += "\t-"
                print(string,file=outfile)

def save_comments(allComments,outfileName):
    """ Prints comments into a file. 
    Inputs: dictionary that contains comments where the keys are the 
    story id and the name of the registered user. Outfile where results 
    will be printed """
    with open(outfileName,"w") as outfile:
        for idName in allComments:
            for writter in allComments[idName]:
                print(idName+"\t"+writter+"\t"+\
                "\t".join(allComments[idName][writter]),file=outfile)

########################################################################
# Basic modules
########################################################################

def create_folder(folderName):
    """Creates a folder if it doesn't exist"""
    if not os.path.exists(folderName):
        cmd = "mkdir "+folderName
        os.system(cmd)

parser = argparse.ArgumentParser(description="Obtain metadata for AO3")
parser.add_argument("-l",dest="linkName",action="store",required=True,\
    help="Link name for the main list of works, put link between \"\" when complex link ")
parser.add_argument("-o",dest="outFolder",action="store",default="results/",\
    help="Folder name where tables will be stored")
parser.add_argument("-c",dest="continueDownload",action="store",default="1",\
    help="Restart point, give last page processed")
parser.add_argument("--base_html",dest="baseHTML",action="store",\
    default="https://archiveofourown.org/",\
    help="Main HTML page for the archive")
parser.add_argument("--download_comments",dest="downComments",\
    action="store_true",help="Downloads comments on top of regular metadata. \
    (Takes a long time due to increasing number of downloads)")
args = parser.parse_args()

""" Base archive link """
baseHTML = args.baseHTML
    
""" Load HTML in BeautifulSoup parsing it with lxml - 
This will always be done for the first page"""
soup = get_webpage(args.linkName)

""" Obtain the total number of pages within the provided link """
number_of_pages = int(str(soup.find_all(class_="pagination actions")[1].\
    find_all("a")[-2]).split(">")[1].split("<")[0])

if number_of_pages == 5000:
    print("WARNING: Be aware that 5000 is the maximum amount of pages\
that the archive creates. To obtain the full list of stories\
apply some filters")

""" Create folders where data is going to be saved """
create_folder(args.outFolder)
create_folder(args.outFolder+"/metadata/")

if args.downComments:
    create_folder(args.outFolder+"/comments/")


""" Obtain metadata from the works of the page """

for pageNum in range(args.continueDownload,number_of_pages):
    if "?" not in args.linkName:
        linkName = args.linkName+"?page="+str(pageNum)
    else:
        linkName = args.linkName+"&page="+str(pageNum)
    
    soup = get_webpage(linkName)
    
    metadata = get_metadata(soup, baseHTML)
    
    
    """ Obtain Comments and Kudos. Kudos can be taken from the 
    first page of the comments page when present so it doesn't require 
    a specific download """
    allKudos = {}
    for idName in metadata:
        if metadata[idName]["author"]:
            if int(metadata[idName]["numKudos"]) != 0:
                soup = get_webpage(metadata[idName]["linkKudos"])
                kudos, metadata[idName]["published_date"] = get_kudos(soup)
                allKudos[idName] = kudos
            else:
                allKudos[idName] = "-"
            
    outfileName = args.outFolder+"/metadata/page"+str(pageNum)+".txt"
    save_metadata(metadata,allKudos,outfileName)
    if args.downComments:
        allComments = {}
        for idName in metadata:
            if int(metadata[idName]["numComments"]) != 0:
                soup = get_webpage(metadata[idName]["linkComments"])
                comments = get_comments(soup,idName,metadata[idName]["author"])
                allComments[idName] = comments
            outfileName = args.outFolder+"/comments/page"+str(pageNum)+".txt"
            save_comments(allComments,outfileName)


