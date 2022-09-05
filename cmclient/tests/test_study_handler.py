from unittest import TestCase

from cmclient.api.study import StudyHandler


class TestStudyHandler(TestCase):

    def test_lifecycle(self):
        handler = StudyHandler()
        handler.handle()

