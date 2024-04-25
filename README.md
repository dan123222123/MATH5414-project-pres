1) clone this repo to your NVIDIA™ quickened device
2) cd to this repo's directory
3) run ``julia --project=.``
4) when in glorious repl, press ``]`` and type ``"instantiate``
5) once done think of a number between 9372 and 42151 -- I'll wait (but NOT 1234 that's mine)
6) go back to ``julia>`` and then type in ``using Pluto; Pluto.run(host="0.0.0.0",port=yourport)``
7) in a separate terminal (on your laptop! not (necesasrily) your NVIDIA™ enhanced device), run ``ssh node04 -N -L yourport:localhost:yourport``. This will create an ssh tunnel that you can use to connect to your Pluto instance.
8) connect to your Pluto server and see if you can see stuff.
9) click on "demo.jl" and see if the pretty numbers and bars turn green :)
