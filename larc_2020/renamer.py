#coding=utf-8

#*IMPORTS
import os

#*REWRITING TEST IMAGES NAME
os.chdir('test_images')
cont = 1

os.system('clear')
print('Stating file renaming...\n')
for image in sorted(os.listdir(os.getcwd())):
  newName = 'robot_test_' + str(cont) + '.jpg'
  os.rename(image, newName)
  print(image + ' remamed to ' + newName)
  cont+=1