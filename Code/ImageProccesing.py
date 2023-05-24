from Detector import *

def getImageTags():
  detector = Detector()
  name = detector.detect("KnowYourWildlife/images/uploaded_image.jpg")
  output = "Tell me something about" + name + "." + '\n' + "Conservation of " + name
  return output 
