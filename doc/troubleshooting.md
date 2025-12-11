# Troubleshooting

## Common Errors and Fixes  

### Error: Anything causing the application to freeze
**Cause**: Nothing happens when clicking on any widget.  
**Fix**: Restart the application, use *Advanced parameters* and *Reset all settings*. Restart the application again.

### Error: "GUI Fails to Launch"  
**Cause**: Wrong python version (e.g., python3.10).  
**Fix**: Reinstall with [python 3.13](https://www.python.org/downloads/release/python-3139/)

### Error: "GUI Fails to Launch"  
**Cause**: Missing dependency (e.g., PyQt5).  
**Fix**: Reinstall with `pip install Cellects`.

### Error: "Please, enter a valid path"  
**Cause**: Incorrect file path in Data Localisation.  
**Fix**: Ensure the path uses forward slashes (`/`) or double backslashes (`\\`).
**Fix**: Ensure the path leads to a directory containing images with the right prefix and extension.

## Debugging Tips  
- Run Cellects in verbose mode:  
```bash
cellects --verbose gui
```
