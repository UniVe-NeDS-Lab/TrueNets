from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, and_
from geoalchemy2.functions import GenericFunction
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape, from_shape
from building import Building_CTR, Building_OSM
from comune import Comune

class BuildingInterface():
    def __init__(self, DSN, srid):
        engine = create_engine(DSN, client_encoding='utf8', echo=False)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.srid = srid

    def get_area(self, name):
        comune = Comune.get_by_name(self.session, name)
        return comune.shape()

    @classmethod
    def get_best_interface(cls, DSN, srid, area_name):
        CTR = CTRInterface(DSN, srid)
        area = CTR.get_area(area_name)
        OSM = OSMInterface(DSN, srid)
        if(CTR.count_buildings(area) > OSM.count_buildings(area)):
            print("Choosed CTR")
            CTR.area = area
            return CTR
        else:
            print("Choosed OSM")
            OSM.area = area
            return OSM


class CTRInterface(BuildingInterface):
    def __init__(self, DSN, srid='4326'):
        super(CTRInterface, self).__init__(DSN, srid)
        self.building_class = Building_CTR


    def get_buildings(self, shape, area=None):
        """Get the buildings intersecting a shape
        point: shapely object
        """
        wkb_element = from_shape(shape, srid=self.srid)
        if area:
            wkb_area = from_shape(area, srid=self.srid)
            building = self.session.query(Building_CTR) \
                .filter(Building_CTR.geom.ST_Intersects(wkb_element),
                        Building_CTR.geom.ST_Intersects(wkb_area)) \
                .order_by(Building_CTR.gid)
        else:
            building = self.session.query(Building_CTR) \
                .filter(Building_CTR.geom.ST_Intersects(wkb_element)) \
                .order_by(Building_CTR.gid)

        return building.all()

    def count_buildings(self, shape):
        """Get the buildings intersecting a shape
        point: shapely object
        """
        wkb_element = from_shape(shape, srid=self.srid)
        building = self.session.query(Building_CTR) \
            .filter(Building_CTR.geom.ST_Intersects(wkb_element))
        return building.count()

    def get_building_gid(self, gid):
        """Get building by gid
        gid: identifier of building
        """
        building = self.session.query(Building_CTR) \
            .filter_by(gid=gid).first()
        return building


class OSMInterface(BuildingInterface):
    def __init__(self, DSN, srid='4326'):
        super(OSMInterface, self).__init__(DSN, srid)
        self.building_class = Building_OSM

    def get_buildings(self, shape, area=None):
        """Get the buildings intersecting a shape
        point: shapely object
        """
        wkb_element = from_shape(shape, srid=self.srid)
        if area:
            wkb_area = from_shape(area, srid=self.srid)
            building = self.session.query(Building_OSM) \
                .filter(and_(Building_OSM.geom.ST_Intersects(wkb_area),
                             Building_OSM.geom.ST_Intersects(wkb_element)))\
                .order_by(Building_OSM.gid)
        else:
            building = self.session.query(Building_OSM) \
                .filter(Building_OSM.geom.ST_Intersects(wkb_element))\
                .order_by(Building_OSM.gid)
        return building.all()


    def count_buildings(self, shape):
        """Get the buildings intersecting a shape
        point: shapely object
        """
        wkb_element = from_shape(shape, srid=self.srid)
        building = self.session.query(Building_OSM) \
            .filter(Building_OSM.geom.ST_Intersects(wkb_element))
        result = building.count()
        return result

    def get_building_gid(self, gid):
        """Get building by gid
        gid: identifier of building
        """
        building = self.session.query(Building_OSM) \
            .filter_by(gid=gid).first()
        return building

    def get_building_osm(self, osm_id):
        building = self.session.query(Building_OSM) \
            .filter_by(osm_id=str(osm_id)).first()
        return building
