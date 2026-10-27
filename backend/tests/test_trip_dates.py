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

    def test_flight_time_is_not_treated_as_calendar_shift(self):
        profile = {"checkin_date": "2026-11-10", "nights": 7}
        self.assertIsNone(update_trip_dates("я хочу вылетать не в 6 утра. поставь самолет попозже", profile))
        self.assertEqual(profile["checkin_date"], "2026-11-10")


    def test_russian_named_month_date(self):
        profile = {"nights": 7}
        self.assertIsNone(update_trip_dates("17 ноября 2026", profile))
        self.assertEqual(profile["checkin_date"], "2026-11-17")
        self.assertEqual(profile["checkout_date"], "2026-11-24")

    def test_russian_named_month_date_range(self):
        profile = {}
        self.assertIsNone(update_trip_dates("с 17 по 24 ноября 2026", profile))
        self.assertEqual(profile["checkin_date"], "2026-11-17")
        self.assertEqual(profile["checkout_date"], "2026-11-24")
        self.assertEqual(profile["nights"], 7)

    def test_shift_with_cart_fallback_succeeds(self):
        profile = {"nights": 7}
        cart = {"checkin_date": "2026-11-15", "checkout_date": "2026-11-22"}
        self.assertIsNone(update_trip_dates("Сдвинь поездку на 2 дня позже", profile, cart))
        self.assertEqual(profile["checkin_date"], "2026-11-17")
        self.assertEqual(profile["checkout_date"], "2026-11-24")


if __name__ == "__main__":
    unittest.main()
