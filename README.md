# 3DBAG dashboard

## Required inputs

- reconstructed_features.csv
- metadata.json
- validate_compressed_files.csv

Use the `convert_to_parquet.py` script to convert the CSV files to parquet, which is the 
required input format for the dashboard.

## How to deploy on Godzilla

1. Upload the code to `/var/www/3dbag-dashboard` and change to the directory 
2. Make a venv with python3.11 in ´./venv´
3. Install all with `pip install -r requirements.txt && pip install .`
4. Read https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-20-04
5. `sudo ln -s /var/www/3dbag-dashboard/3dbag_dashboard.service /etc/systemd/system/3dbag_dashboard.service`
6. `sudo systemctl start 3dbag_dashboard && sudo systemctl enable 3dbag_dashboard`
7. `sudo systemctl status 3dbag_dashboard`
8. Add the dashboard path location to the nginx server
    ```
            location /dashboard/ {
                include proxy_params;
                proxy_pass http://unix:/var/www/3dbag-dashboard/app/3dbag_dashboard.sock;
            }
    ```
