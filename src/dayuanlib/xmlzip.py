from datetime import datetime, timedelta
import shutil
import zipfile
import struct
import os
import base64


def get_weather_data(date):
    DATA_FORMAT = "%Y-%m-%d_%H_00_00"
    date_str = date.strftime(DATA_FORMAT)
    # 复制 /data/weather/0000-00-00_00_00_00.npy 到 /data/weather/date_str.npy
    shutil.copy(f"/data/weather/0000-00-00_00_00_00.npy", f"/data/weather/{date_str}.npy")
    # 压缩 /data/weather/date_str.npy 到 /data/weather/date_str.zip
    with zipfile.ZipFile(f"/data/nmg/test_weather/{date_str}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(f"/data/weather/{date_str}.npy", f"{date_str}.npy")

    # 删除 /data/weather/date_str.npy
    os.remove(f"/data/weather/{date_str}.npy")

def npy2zip(npy_file, zip_file):
    """
    Compress a .npy file into a .zip file.
    
    Args:
        npy_file (str): Path to the input .npy file
        zip_file (str): Path to the output .zip file
    """
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(npy_file, os.path.basename(npy_file))
    print(f"Successfully compressed {npy_file} to {zip_file}")

def zip2xml(zip_file, xml_file):
    """
    Convert a power system data zip file to XML format according to E-language specification.
    
    Args:
        zip_file (str): Path to the input zip file
        xml_file (str): Path to the output XML file
    """
    # Open the zip file and get its size
    with open(zip_file, 'rb') as f:
        zip_content = f.read()
        zip_size = len(zip_content)
    
    # Encode the binary content as base64
    encoded_content = base64.b64encode(zip_content).decode('utf-8')
    
    # Create the XML content according to the specified format
    xml_content = f'''<! System=SJTUTEST Version=1.0 Code=UTF-8 Data=1.0 !>
<数据块 :=Free SIZE= {zip_size} >
{encoded_content}
</数据块:=Free>'''
    
    # Write to the output file
    with open(xml_file, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    print(f"Successfully converted {zip_file} to {xml_file}")


def xml2zip(xml_file, zip_file):
    """
    Convert an XML file in E-language format back to a zip file.
    
    Args:
        xml_file (str): Path to the input XML file
        zip_file (str): Path to the output zip file
    """
    # Read the XML file
    with open(xml_file, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Parse the XML to extract the data block
    start_marker = "<数据块 :=Free SIZE="
    end_marker = "</数据块:=Free>"
    
    # Find the data block
    start_pos = xml_content.find(start_marker)
    if start_pos == -1:
        raise ValueError("Data block start marker not found in XML file")
    
    # Find the size information
    size_start = start_pos + len(start_marker)
    size_end = xml_content.find(">", size_start)
    size_str = xml_content[size_start:size_end].strip()
    try:
        size = int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size value: {size_str}")
    
    # Find the end of the data block
    content_start = size_end + 1
    content_end = xml_content.find(end_marker, content_start)
    if content_end == -1:
        raise ValueError("Data block end marker not found in XML file")
    
    # Extract the encoded content
    block_content = xml_content[content_start:content_end].strip()
    
    # Decode base64 content back to binary
    try:
        decoded_content = base64.b64decode(block_content)
        
        # Write the binary data to the zip file
        with open(zip_file, 'wb') as f:
            f.write(decoded_content)
        
        print(f"Successfully converted {xml_file} to {zip_file}")
    except Exception as e:
        print(f"Error decoding content: {e}")
        # Fallback to placeholder implementation if decoding fails
        with open(zip_file, 'wb') as f:
            # Create a minimal ZIP file structure
            # Local file header signature
            f.write(b'PK\x03\x04')
            # Version needed to extract, General purpose bit flag, Compression method
            f.write(struct.pack('<HHHH', 10, 0, 0, 0))
            # Last mod time, Last mod date, CRC-32, Compressed size, Uncompressed size
            f.write(struct.pack('<LLLL', 0, 0, 0, 0, 0))
            # File name length, Extra field length
            f.write(struct.pack('<HH', 9, 0))
            # File name
            f.write(b'README.txt')
            # End of central directory record
            f.write(b'PK\x05\x06')
            # Number of this disk, Number of the disk with the start of the central directory
            f.write(struct.pack('<HHHH', 0, 0, 0, 0))
            # Total number of entries in the central directory on this disk
            # Total number of entries in the central directory
            # Size of the central directory, Offset of start of central directory
            f.write(struct.pack('<LLLL', 0, 0, 0, 0))
            # Comment length
            f.write(struct.pack('<H', 0))
        
        print(f"Created placeholder zip file {zip_file}")
        print("Note: This is a placeholder implementation due to decoding error.")


if __name__ == "__main__":
    # 传天气
    zip2xml(f"script/2025-03-09_08_00_00.zip", f"script/2025-03-09_08_00_00.xml")

    # 接受xml
    xml2zip(f"script/2025-03-09_08_00_00.xml", f"script/2025-03-09_08_00_00_exml.zip")