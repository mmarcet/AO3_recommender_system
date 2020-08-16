#!/usr/bin/env python

#Clean metadata obtained from the AO3 scrapper

with open("metadata_fics.cleaned.txt","w") as outfile:
    #Print header
    header = [x.strip() for x in open("header.txt")][0]
    idNames = set([])
    print(header,file=outfile)
    with open("metadata_fics.txt","r") as infile:
        for line in infile:
            line = line.strip()
            dades = line.split("\t")
            if dades[0] not in idNames:
                idNames.add(dades[0])
                num_unknown_fields = dades.count("-")
                if num_unknown_fields > 3:
                    pass
                else:
                    #Replace , in numbers
                    dades[6] = dades[6].replace(",","")
                    try:
                        numWords = int(dades[6])
                    except:
                        print("check:",dades[0],num_unknown_fields)
                    #Add - to empty fields
                    for a in range(len(dades)):
                        if dades[a] == "":
                            dades[a] = "-"
                    #Remove spaces so that each tag is considered as one 
                    #entity then substitute | in tags, characters and 
                    #relationships for spaces so that they can be interpreted 
                    #as different words
                    dades[11] = dades[11].replace(" ","").replace("/","").replace("|"," ")
                    dades[12] = dades[12].replace(" ","").replace("|"," ")
                    dades[13] = dades[13].replace(" ","").replace("|"," ")
                    print("\t".join(dades),file=outfile)
