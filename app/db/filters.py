from qdrant_client.http import models


def build_geo_fileter(lat: float, lon: float, radius_m: int = 1000) -> models.Filter:
    """
    중심 좌표(lat,lon)로 부터 반경 내의 데이터만 필터링 하는 조건 생성
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key="location",
                geo_radius=models.GeoRadius(
                    center=models.GeoPoint(lat=lat, lon=lon), radius=radius_m
                ),
            ),
        ]
    )


def build_geo_fileter_with_content_type(
    lat: float, lon: float, radius_m: int = 1000, content_type: str = "관광지"
) -> models.Filter:
    """
    중심 좌표(lat,lon)로 부터 반경 내의 데이터만 필터링 하는 조건 생성
    """
    return build_geo_fileter(lat, lon, radius_m).must.append(
        models.FieldCondition(key="content_type", value=content_type)
    )