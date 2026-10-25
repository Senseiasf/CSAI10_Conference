import unittest

from app.services.trip_dates import update_trip_dates


class TripDatesTest(unittest.TestCase):
    def test_calendar_stay_crosses_year(self):
        profile = {"nights": 7}
        self.assertIsNone(update_trip_dates("заезд 28.12.2026", profile))
        self.assertEqual(profile["checkout_date"], "2027-01-04")

    def test_shift_keeps_duration_and_moves_both_dates(self):
        for message, expected in (("на 2 дня позже", "2026-11-12"),
                                  ("на 3 дня раньше", "2026-11-07"),
                                  ("раньше на 4 дня", "2026-11-06")):
            with self.subTest(message=message):
                profile = {"checkin_date": "2026-11-10", "nights": 7}
                self.assertIsNone(update_trip_dates(message, profile))
                self.assertEqual(profile["checkin_date"], expected)
                self.assertEqual(profile["nights"], 7)
                self.assertEqual(int(profile["checkout_date"][-2:]), int(expected[-2:]) + 7)

    def test_shift_without_base_date_asks_question(self):
        profile = {"nights": 7}
        self.assertIsNotNone(update_trip_dates("на 2 дня позже", profile))
        self.assertEqual(profile, {"nights": 7})

    def test_duration_updates_checkout(self):
        profile = {"checkin_date": "2026-11-10", "nights": 7}
        self.assertIsNone(update_trip_dates("на 4 ночи", profile))
        self.assertEqual(profile["checkin_date"], "2026-11-10")
        self.assertEqual(profile["checkout_date"], "2026-11-14")

    def test_month_without_day_needs_clarification(self):
        profile = {"checkin_date": "2026-11-10", "nights": 7}
        self.assertIsNotNone(update_trip_dates("а если в декабре?", profile))
        self.assertEqual(profile["checkin_date"], "2026-11-10")

    def test_invalid_date_does_not_replace_previous(self):
        profile = {"checkin_date": "2026-11-10", "nights": 7}
        self.assertIsNotNone(update_trip_dates("31.02.2026", profile))
        self.assertEqual(profile["checkin_date"], "2026-11-10")

    def test_iso_and_duration_can_be_set_together(self):
        profile = {}
        self.assertIsNone(update_trip_dates("2028-02-28 на 3 ночи", profile))
        self.assertEqual(profile["checkout_date"], "2028-03-02")

    def test_duration_outside_supported_range_is_not_silently_ignored(self):
        profile = {"nights": 7}
        self.assertIsNotNone(update_trip_dates("на 0 ночей", profile))
        self.assertEqual(profile["nights"], 7)


if __name__ == "__main__":
    unittest.main()
