# 2P imaging pipeline

Using suite2p. 
Based on code from Johannes Kappel. Classifier developed by Enrico Kohn and Katja Slangewal. Cellpose model from Inbal
Shainer. BiDiOffset function from Joseph Donovan. 

This is supposed to be a analysis pipeline usable by other people, of 2p imaging data. 
I want people to be able to plug and play, with minimal parameter setting, their data. 
Specicially, it is for analysis over multiple experiments, that have to be suite2p'ed together, but then have F traces
seperately. Also, with flexibility over several planes. Also, with ability to hash a specific run. 

So It has processing unit the daughter class, inherited by suite2prun unit. 
Does that make the most sens?


What people need to know:
Which animal numbers? 
Which dates?
Which experiment numbers?
Which experiment lengths?
Raw data path?
processed data path?

framerate and nplanes

start iwth config setup
# Generally, you should only run together the data that has the same image acquisition.
# E.g. framerate/number planes/resolution.
# Because this defines your suite2p settings.