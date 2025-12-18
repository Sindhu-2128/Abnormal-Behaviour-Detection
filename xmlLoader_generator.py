import lxml.etree as et
import os

class Poi_handle():
    def __init__(self, path=None):
        if path is None:
            raise ValueError("A path to 'poi.xml' must be provided.")
        
        self.path = path
        parser = et.XMLParser(remove_blank_text=True)
        
        if os.path.exists(path):
            self.tree = et.parse(path, parser)
        else:
            print(f"Warning: XML file not found at {path}. Calibration will fail.")
            root = et.Element("root")
            self.tree = et.ElementTree(root)

    def searchPic(self, n):
        return self.tree.getroot().find(f".//pic[@n='{str(n).zfill(3)}']")

    def add(self, n, y, *xs):
        root = self.tree.getroot()
        picPt = root.find(f"pic[@n='{str(n)}']")
        if picPt is None:
            picPt = et.SubElement(root, "pic", n=str(n))

        yPt = picPt.find(f"y[@val='{str(y)}']")
        if yPt is None:
            yPt = et.SubElement(picPt, 'y', val=str(y))

        for child in list(yPt):
            yPt.remove(child)
        for x in xs:
            et.SubElement(yPt, "point").text = str((x, y))

        self.tree.write(self.path, pretty_print=True, xml_declaration=True, encoding='utf-8')