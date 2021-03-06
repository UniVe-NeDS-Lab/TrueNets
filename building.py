from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String,  Unicode
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape
import numpy as np
from geoalchemy2.functions import ST_Intersects

Base = declarative_base()


class Building(Base):
    __abstract__ = True
    gid = Column(Integer, primary_key=True)
    geom = Column(Geometry('POLYGON'))
    height = 4

    def __hash__(self):
        return hash(self.gid)

    def __eq__(self, other):
        if type(other) == int:
            return other == self.gid
        else:
            # equality performed only on gid (unique in the db)
            return self.gid == other.gid
    
    def __lt__(self, other):
        return self.gid < other.gid
    
    def __gt__(self, other):
        return self.git > other.gid

    def __repr__(self):
        return str(self.gid)

    def shape(self):
        return to_shape(self.geom)

    def coords(self):
        return self.shape().representative_point()

    def xy(self):
        return (self.coords().x, self.coords().y)

    def coord_height(self):
        obj = {}
        obj['coords'] = self.coords()
        obj['height'] = self.height
        obj['building'] = self
        return obj


class Building_CTR(Building):
    __tablename__ = 'ctr'

    def id(self):
        return self.gid

    def __str__(self):
        return "Building ID: {0} \nLongitude: {1} \nLatitude: {2} \nCodice: {3}"\
               .format(self.gid, self.coords().x, self.coords().y, self.codice)


class Building_OSM(Building):
    __tablename__ = 'osm_buildings_3003'
    osm_id = Column(Integer)
    code = Column(Integer)
    fclass = Column(String)
    name = Column(Unicode)
    t_type = Column('type', String)

    def id(self):
        return self.osm_id

    def __repr__(self):
        return str(self.gid)

    def __str__(self):
        if(self.name):
            return "Name: {3} \nBuilding ID: {0} \nLongitude: {1} \nLatitude: {2}"\
                .format(self.gid, self.coords().x, self.coords().y, self.name)
        return "Building ID: {0} \nLongitude: {1} \nLatitude: {2}"\
            .format(self.gid, self.coords().x, self.coords().y)
