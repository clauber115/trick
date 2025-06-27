trick.real_time_enable()
trick.exec_set_software_frame(0.010)
trick.itimer_enable()
trick.exec_set_rt_nap(1)
trick.frame_log_on()
print("NAP? ",trick.exec_get_rt_nap())

trick.exec_set_enable_freeze(True)
#trick.exec_set_freeze_command(True)

#simControlPanel = trick.SimControlPanel()
#trick.add_external_application(simControlPanel)
