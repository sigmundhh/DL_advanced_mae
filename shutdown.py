import os
import sys

def shutdown(mins=5):
  os.system("sudo /usr/sbin/shutdown " + str(mins) +" poweroff scheduled")

def shutdown_cancel():
  print("Cancelling shutdown...")
  os.system("sudo /usr/sbin/shutdown -c")

if len(sys.argv) == 1:
  print("Executing default behavior: shutting down in 5 minutes")
  shutdown()
elif len(sys.argv) == 2:
  if "-c" in sys.argv or "--cancel" in sys.argv:
    shutdown_cancel()
  else:
    try:
      mins = int(sys.argv[1])
      shutdown(mins)
    except Exception:
      print("Erroneous argument argument: whole number of minutes until poweroff or -c/--cancel  expected")
