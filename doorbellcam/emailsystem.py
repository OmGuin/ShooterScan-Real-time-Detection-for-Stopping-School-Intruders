import smtplib


email = "om.guin@gmail.com"

reciever_email = "pranav.sambhu@gmail.com"




subject = "[URGENT] GUN DETECTED ON CAMPUS"
message = 'A FIREARM HAS BEEN DETECTED BY A CCTV CAMERA ON CAMPUS. THIS IS NOT A TEST. LOCK DOWN IMMEDIATELY'

text = f"Subject: {subject}\n\n{message}"

server = smtplib.SMTP("smtp.gmail.com", 587)

server.starttls()

server.login(email, "enor kisf wbzp jjkf")

server.sendmail(email, reciever_email, text)
print("Email sent")