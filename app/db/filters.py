from qdrant_client.http import models
from typing import Optional


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
    lat: float, lon: float, radius_m: int = 1000, content_type: Optional[str] = "관광지"
) -> models.Filter:
    """
    중심 좌표(lat,lon)로 부터 반경 내 + 특정 컨텐츠 타입 데이터만 필터링
    """
    # 1. 기본 지리적 필터 생성
    geo_filter = build_geo_fileter(lat, lon, radius_m)
    if content_type:
        # 2. content_type 조건 생성 (Qdrant 문법 준수: match 사용)
        content_condition = models.FieldCondition(
            key="content_type", match=models.MatchValue(value=content_type)
        )

        # 3. 기존 필터의 must 리스트에 추가
        # (append는 반환값이 None이므로, return 문에서 바로 쓰면 안 됩니다)
        geo_filter.must.append(content_condition)

    # 4. 수정된 필터 객체 반환
    return geo_filter
