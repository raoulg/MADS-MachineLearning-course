# GUI

## Picking an interface
Picking an interface is a really broad topic. It really depends on your usecase:

- Does your client has a certain environment / requirements for the endpoint?
- If not explicit, how technical is your client? 

Examples I have seen:

- **<ins>The cronjob + email</ins>** : Every thursday, data is dumped in the cloud. A cronjob kicks off and starts a FTP connection to pull the data, a model is executed, the results is converted to excel, and the client gets an email with the excel as an attachement.
- **<ins>CLI</ins>**: A command line interface with [gum](https://github.com/charmbracelet/gum). If your client technical enough to start a terminal, this can offer a nice interactive script.
![](https://camo.githubusercontent.com/f820a22f7574d55e1d9ccd3bfb0d8c337811ff05ea07d8d4b504dd0dc09ee24e/68747470733a2f2f73747566662e636861726d2e73682f67756d2f64656d6f2e676966)
- **<ins>minimal dashboard</ins>**: A [streamlit](https://streamlit.io/) dashboard. You can build it in python, and it is pretty quick to setup. Downside: it doestn scale well (e.g. using it with 20 people at the same time will be problematic) and it is hard to control how things look. Upside: it is fast, and the default looks good.
- **<ins>Complete dashboard</ins>**: A complete dashboard. When using Julia, you can set up a complete dashboard within half an hour, see [this video](https://www.youtube.com/watch?v=YEQLTCWxDuM). You have HTML drag-and-drop, and your website can more easily scale. Another option is [flask](https://flask.palletsprojects.com/en/2.2.x/quickstart/) which is python, but setting up a full website is a bit more complex. 
- **<ins>API</ins>**: You spin up an api (eg with [fastapi](https://fastapi.tiangolo.com/tutorial/first-steps/)) and another process can send a request in JSON, and receive an answer (e.g. for a website in production). This is often ideal for frontend developpers: the know how the send a request in JSON, and they know how to put the JSON back into the website to serve to the client. 
