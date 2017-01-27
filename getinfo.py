from pycuda import compiler, driver
from pycuda import autoinit

'''
# Print all attributes
for devicenum in range(driver.Device.count()):
	device=driver.Device(devicenum)
	attrs=device.get_attributes()

	#Beyond this point is just pretty printing
	print("\n===Attributes for device %d"%devicenum)
	for (key,value) in attrs.iteritems():
		print("%s:%s"%(str(key),str(value)))
'''
# Print one attribute
#device=driver.Device(0)
#attrs=device.get_attributes()
#print attrs[driver.device_attribute.MAX_THREADS_PER_BLOCK]

# Print one attribute
print driver.Device(0).get_attributes()[driver.device_attribute.MAX_THREADS_PER_BLOCK]