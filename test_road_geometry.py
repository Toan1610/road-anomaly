import unittest
import numpy as np
from road_geometry import (
    reconstruct_corridor, uncertain_road_surface, expand_camera_foreground,
    camera_foreground_neighborhood,
    has_obstacle_support,
)


class RoadGeometryTests(unittest.TestCase):
    def scene(self):
        road=np.zeros((240,400),np.uint8)
        for y in range(40,231):
            road[y,20+y//5:360-y//12]=1
        return road

    def test_visible_asymmetric_curbs_do_not_move_to_template_or_prior(self):
        road=self.scene()
        result=reconstruct_corridor(road,np.zeros_like(road),40,230,
                                    np.full(191,150),np.full(191,250))
        left,right=result[5:]
        for j,y in enumerate(range(40,231)):
            self.assertEqual(left[j],np.flatnonzero(road[y])[0])
            self.assertEqual(right[j],np.flatnonzero(road[y])[-1])

    def test_occluded_curb_interpolates_between_observed_rows(self):
        road=self.scene(); obstacle=np.zeros_like(road)
        road[90:160,:140]=0;obstacle[90:160,20:141]=1
        meta={}
        result=reconstruct_corridor(road,obstacle,40,230,diagnostics=meta)
        self.assertLess(abs(result[3][120]-(20+120//5)),3)
        self.assertFalse(meta['observed_left'][120-40])
        self.assertTrue(meta['observed_left'][200-40])

    def test_no_road_never_creates_white_trapezoid(self):
        empty=np.zeros((240,400),np.uint8)
        result=reconstruct_corridor(empty,empty,40,230)
        self.assertEqual(np.count_nonzero(result[0]),0)

    def test_unbounded_occlusion_does_not_extrapolate_across_image(self):
        road=self.scene();obstacle=np.zeros_like(road)
        obstacle[:160]=1
        road[:90]=0
        result=reconstruct_corridor(road,obstacle,40,230)
        self.assertFalse(result[0][:90].any())
        self.assertEqual(result[3][120],20+120//5)
        self.assertEqual(result[1][0,1],90)

    def test_confident_road_is_not_relabelled_by_appearance(self):
        frame=np.full((100,200,3),150,np.uint8)
        pred=np.zeros((100,200),np.uint8)
        score=np.ones((100,200),np.float32)
        self.assertFalse(uncertain_road_surface(frame,pred,score).any())

    def test_camera_roi_does_not_grow_to_opposite_material_pile(self):
        pred=np.zeros((300,500),np.uint8)
        pred[210:,330:]=3;pred[210:,:110]=3
        initial=np.zeros_like(pred);initial[260:,390:]=255
        expanded=expand_camera_foreground(initial,pred)
        cached=expand_camera_foreground(initial,pred,neighborhood=camera_foreground_neighborhood(initial))
        np.testing.assert_array_equal(expanded,cached)
        self.assertTrue(expanded[250,380])
        self.assertFalse(expanded[250,50])

    def test_dashboard_strip_does_not_hide_side_camera_seed(self):
        pred=np.zeros((300,500),np.uint8);pred[210:,330:]=3
        initial=np.zeros_like(pred);initial[260:,390:]=255;initial[290:]=255
        expanded=expand_camera_foreground(initial,pred)
        self.assertTrue(expanded[250,380])
        self.assertFalse(expanded[250,50])

    def test_curb_band_needs_surface_interior_or_enclosed_object(self):
        component=np.ones((40,100),bool);empty=np.zeros_like(component)
        self.assertFalse(has_obstacle_support(component,empty,empty))
        core=empty.copy();core[10:30,20:80]=True
        self.assertTrue(has_obstacle_support(component,core,empty))
        self.assertTrue(has_obstacle_support(component,empty,component))

    def test_only_weak_road_connected_to_material_seed_is_removed(self):
        frame=np.full((100,200,3),150,np.uint8)
        pred=np.zeros((100,200),np.uint8);score=np.ones((100,200),np.float32)
        pred[20:80,:30]=3;score[20:80,:30]=.01;score[20:80,30:70]=.3
        score[20:80,120:150]=.3
        result=uncertain_road_surface(frame,pred,score)
        self.assertTrue(result[40,50])
        self.assertFalse(result[40,130])
        blocked=np.zeros_like(pred);blocked[:,:80]=255
        self.assertFalse(uncertain_road_surface(frame,pred,score,exempt=blocked).any())


if __name__=='__main__':
    unittest.main()
