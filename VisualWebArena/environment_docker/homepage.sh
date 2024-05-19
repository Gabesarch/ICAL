# Define your actual server hostname
YOUR_ACTUAL_HOSTNAME="http://gsarch@tayer.psy.cmu.edu"
# Remove trailing / if it exists
YOUR_ACTUAL_HOSTNAME=${YOUR_ACTUAL_HOSTNAME%/}
# Use sed to replace placeholder in the HTML file
perl -pi -e "s|http://gsarch@tayer.psy.cmu.edu|${YOUR_ACTUAL_HOSTNAME}|g" webarena-homepage/templates/index.html

cd webarena-homepage
flask run --host=0.0.0.0 --port=4399