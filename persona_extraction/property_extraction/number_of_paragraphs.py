def count_paragraphs(text):
    # Split the text into paragraphs based on newline characters
    paragraphs = text.split('\n\n')
    # Remove empty paragraphs
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    # Count the number of paragraphs
    num_paragraphs = len(paragraphs)
    return num_paragraphs

def main():
    text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Sed feugiat urna in velit vestibulum, et rutrum diam gravida. 
    Proin in lacus nec leo mattis malesuada. 

    Nullam nec justo id turpis accumsan rutrum. 
    Sed tempor urna eget lacus volutpat, vel pretium ligula malesuada. 
    Fusce convallis justo eu nisi varius aliquam."""
    
    num_paragraphs = count_paragraphs(text)
    
    print({"Number of paragraphs": num_paragraphs})

if __name__ == "__main__":
    main()
