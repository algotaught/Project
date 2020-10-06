'''importing the datetime library for logging the time and the date of the file operations'''

from datetime import datetime


'''creating the function which will take the log message 
and save it in the logger file along with date and time
'''
class log_creator:
    def __init__(self):
        pass

    def log(self,file,message):


        self.file=file
        self.message=message



        now=datetime.now()
        date=now.date()
        time=now.strftime('%H:%M:%S')
        self.file.write(str(date)+': '+str(time+': '+ self.message + '\n')